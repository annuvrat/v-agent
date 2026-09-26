import logging
import math
import os
from pathlib import Path

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    TurnHandlingOptions,
    cli,
    room_io,
    text_transforms,
)
from livekit.plugins import cartesia, deepgram, noise_cancellation, openai, silero

logger = logging.getLogger("voice-agent")

load_dotenv()

SLOW_LLM_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.8"))
STT_LANGUAGE = os.getenv("STT_LANGUAGE", "en-US")
STT_KEYTERMS = [k.strip() for k in os.getenv("STT_KEYTERMS", "").split(",") if k.strip()]
CARTESIA_VOICE = os.getenv("CARTESIA_VOICE", "9626c31c-bec5-4cca-baa8-f8ba9e84c8bc")
# Sonic-3 accepts a free-form emotion descriptor (only the first is used by the
# plugin, so keep it a single rich phrase rather than a list).
CARTESIA_EMOTION = os.getenv("CARTESIA_EMOTION") or "curious, warm and friendly, light and conversational"
# Speech rate as a float (sonic-3 valid range 0.6 - 2.0). Slightly slower than
# 1.0 reads more thoughtful and human when combined with an expressive emotion.
CARTESIA_SPEED = float(os.getenv("CARTESIA_SPEED", "0.94"))
LOG_SUMMARY_EVERY = int(os.getenv("LOG_SUMMARY_EVERY", "10") or "10")

# Use LiveKit's built-in semantic turn detector ("end of turn" model).
# It ships with livekit-agents (>=1.6.1); if unavailable on your version we
# simply fall back to the session's automatic turn detection.
try:
    from livekit.agents.inference import TurnDetector

    TURN_DETECTOR = TurnDetector()
except ImportError:  # pragma: no cover
    TURN_DETECTOR = None


def _configure_logging() -> None:
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    log_file = os.getenv("LOG_FILE", "").strip()
    fmt = "%(asctime)s.%(msecs)03d %(levelname)-7s %(name)s: %(message)s"
    datefmt = "%H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt)

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path, encoding="utf-8"))
    for handler in handlers:
        handler.setLevel(level)
        handler.setFormatter(formatter)

    # attach our own handlers to this logger only and stop propagation, otherwise
    # livekit's root config prints every line a second time in its own format
    logger.handlers.clear()
    for handler in handlers:
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False

    # keep the noise from the SDK itself down unless we're debugging
    for name in ("livekit", "livekit.agents", "livekit.plugins"):
        logging.getLogger(name).setLevel(logging.DEBUG if level_name == "DEBUG" else logging.WARNING)

    logger.info(
        "qa logging enabled level=%s file=%s summary_every=%s",
        level_name,
        log_file or "(terminal only)",
        LOG_SUMMARY_EVERY,
    )


def _fmt_ms(value: float | None) -> str:
    return f"{value * 1000:.0f}ms" if value is not None else "-"


def _fmt_tps(value: float | None) -> str:
    return f"{value:.1f} tok/s" if value is not None else "-"


def _pct(values: list[float], p: float) -> float:
    s = sorted(values)
    k = (len(s) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] + (s[c] - s[f]) * (k - f)


class TurnTracker:
    """Per-turn latency / QA instrumentation driven by AgentSession events.

    Each log line is prefixed with the turn number so a conversation can be
    reconstructed and issues (high latency, cut-off replies) spotted at a glance.
    """

    def __init__(self) -> None:
        self.turn = 0
        self._user_final_at: float | None = None
        self._think_start: float | None = None
        self._speak_start: float | None = None
        self._last_usage_sig: str | None = None

        self._e2e: list[float] = []
        self._ttft: list[float] = []
        self._ttfs: list[float] = []
        self._tts_ttfb: list[float] = []
        self._tps: list[float] = []
        self._speak_durations: list[float] = []

        self.count_interrupted = 0
        self.count_false_interruption = 0
        self.count_crosstalk = 0
        self.count_timeouts = 0
        self.count_errors = 0
        self.user_words = 0

    # -- helpers ---------------------------------------------------------

    def _ms_since(self, now: float, start: float | None) -> float:
        return 0.0 if start is None else (now - start) * 1000

    @staticmethod
    def _word_count(text: str) -> int:
        return len(text.split())

    # -- session event handlers -----------------------------------------

    def on_user_transcript(self, ev) -> None:
        if not ev.is_final:
            return
        self._user_final_at = ev.created_at
        logger.info("TURN#%d USER_FINAL text=%r", self.turn, ev.transcript.strip())

    def on_user_timeout(self, ev) -> None:
        self.count_timeouts += 1
        logger.warning(
            "TURN#%d USER_NOISY speech=%.2fs (no transcript - likely background noise)",
            self.turn,
            ev.speech_duration,
        )

    def on_agent_state(self, ev) -> None:
        now = ev.created_at
        old, new = ev.old_state, ev.new_state
        if new == "thinking":
            self._think_start = now
            logger.info(
                "TURN#%d AGENT state=thinking (%.0fms after user final)",
                self.turn,
                self._ms_since(now, self._user_final_at),
            )
        elif new == "speaking":
            delay = self._ms_since(now, self._user_final_at)
            think = self._ms_since(now, self._think_start)
            logger.info(
                "TURN#%d AGENT state=speaking (%.0fms after user final, thought %.0fms)",
                self.turn,
                delay,
                think,
            )
            self._speak_start = now
        elif old == "speaking":
            dur = self._ms_since(now, self._speak_start)
            self._speak_durations.append(dur / 1000)
            logger.info("TURN#%d AGENT state=stopped_spoke dur=%.0fms", self.turn, dur)

    def on_conversation_item(self, ev) -> None:
        item = getattr(ev, "item", None)
        if item is None or not hasattr(item, "role"):
            return
        role = getattr(item, "role", None)
        text = (getattr(item, "text_content", None) or "").strip()
        metrics = getattr(item, "metrics", None) or {}

        if role == "user":
            self.turn += 1
            self._user_final_at = getattr(item, "created_at", None) or self._user_final_at
            words = self._word_count(text)
            self.user_words += words
            logger.info(
                "TURN#%d USER committed words=%d trans_delay=%s eot_delay=%s text=%r",
                self.turn,
                words,
                _fmt_ms(metrics.get("transcription_delay")),
                _fmt_ms(metrics.get("end_of_turn_delay")),
                text,
            )
            if LOG_SUMMARY_EVERY > 0 and self.turn % LOG_SUMMARY_EVERY == 0:
                self.print_summary(f"every{LOG_SUMMARY_EVERY}")
        elif role == "assistant":
            interrupted = bool(getattr(item, "interrupted", False))
            words = self._word_count(text)

            e2e = metrics.get("e2e_latency")
            if e2e is not None:
                self._e2e.append(e2e)
            spoke = self._ms_since(ev.created_at, self._speak_start)

            for key, bucket in (
                ("llm_node_ttft", self._ttft),
                ("llm_node_ttfs", self._ttfs),
                ("tts_node_ttfb", self._tts_ttfb),
                ("llm_node_tps", self._tps),
            ):
                value = metrics.get(key)
                if value is not None:
                    bucket.append(value)

            if interrupted:
                self.count_interrupted += 1
                logger.warning(
                    "TURN#%d AGENT_INTERRUPTED stopped_midway=True spoke=%.0fms full_words=%d text=%r",
                    self.turn,
                    spoke,
                    words,
                    text,
                )

            logger.info(
                "TURN#%d AGENT_TURN words=%d interrupted=%s e2e=%s llm_ttft=%s llm_ttfs=%s "
                "tps=%s tts_ttfb=%s playback=%s",
                self.turn,
                words,
                interrupted,
                _fmt_ms(e2e),
                _fmt_ms(metrics.get("llm_node_ttft")),
                _fmt_ms(metrics.get("llm_node_ttfs")),
                _fmt_tps(metrics.get("llm_node_tps")),
                _fmt_ms(metrics.get("tts_node_ttfb")),
                _fmt_ms(metrics.get("playback_latency")),
            )

    def on_speech_created(self, ev) -> None:
        handle = getattr(ev, "speech_handle", None)
        logger.info(
            "TURN#%d SPEECH_CREATED id=%s source=%s user_initiated=%s",
            self.turn,
            getattr(handle, "id", "-"),
            getattr(ev, "source", "-"),
            getattr(ev, "user_initiated", False),
        )

    def on_false_interruption(self, ev) -> None:
        self.count_false_interruption += 1
        logger.warning("TURN#%d FALSE_INTERRUPTION resumed=%s", self.turn, ev.resumed)

    def on_overlapping_speech(self, ev) -> None:
        self.count_crosstalk += 1
        logger.warning("TURN#%d CROSSTALK user spoke over agent", self.turn)

    def on_error(self, ev) -> None:
        self.count_errors += 1
        source = ev.source
        logger.error(
            "ERROR source=%s model=%s provider=%s err=%r",
            type(source).__name__ if source is not None else "-",
            getattr(source, "model", None),
            getattr(source, "provider", None),
            ev.error,
        )

    def on_usage(self, ev) -> None:
        parts = []
        for usage in ev.usage.model_usage:
            dump = getattr(usage, "model_dump", None)
            fields = ""
            if dump is not None:
                fields = ", ".join(
                    f"{k}={v}"
                    for k, v in dump().items()
                    if v not in (None, 0, 0.0) and k not in ("type", "provider", "model")
                )
            parts.append(f"{getattr(usage, 'provider', '-')}/{getattr(usage, 'model', '-')}({fields})")
        line = " | ".join(parts)
        # session_usage_updated fires continuously while STT/TTS stream; only log
        # when the figures actually changed so the TURN# lines stay readable.
        if line and line != self._last_usage_sig:
            self._last_usage_sig = line
            logger.info("USAGE %s", line)

    # -- summary ---------------------------------------------------------

    def _stats(self, values: list[float], formatter=_fmt_ms) -> str:
        if not values:
            return "-"
        avg = sum(values) / len(values)
        return (
            f"avg={formatter(avg)} med={formatter(_pct(values, 0.5))} "
            f"p95={formatter(_pct(values, 0.95))} n={len(values)}"
        )

    def print_summary(self, why: str) -> None:
        logger.info(
            "QA-SUMMARY[%s] turns=%d interrupted=%d false_intr=%d crosstalk=%d noisy=%d errors=%d "
            "user_words=%d",
            why,
            self.turn,
            self.count_interrupted,
            self.count_false_interruption,
            self.count_crosstalk,
            self.count_timeouts,
            self.count_errors,
            self.user_words,
        )
        logger.info(
            "QA-SUMMARY[%s] e2e=%s llm_ttft=%s llm_ttfs=%s",
            why,
            self._stats(self._e2e),
            self._stats(self._ttft),
            self._stats(self._ttfs),
        )
        logger.info(
            "QA-SUMMARY[%s] tts_ttfb=%s tps=%s speak_dur=%s",
            why,
            self._stats(self._tts_ttfb),
            self._stats(self._tps, formatter=_fmt_tps),
            self._stats(self._speak_durations),
        )


class VoiceAgent(Agent):
    """The agent's personality / system prompt lives here."""

    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are Mia, a friendly and curious voice assistant. You are heard "
                "over voice, so you talk the way a real person talks - never like a "
                "written document or a scripted bot. "
                "Keep it casual, warm, and conversational. Use short, varied sentences "
                "with a natural rhythm, and use contractions naturally ('that's', "
                "'you're', 'it's'). Let the words breathe: a comma or an occasional "
                "ellipsis (...) creates a natural pause where a human would pause. "
                "Write only plain spoken text - no emojis, no markdown, no bullets, "
                "no numbers, no headings, no headers. Typically answer in one to three "
                "sentences unless the user asks for detail. If you did not hear the "
                "user clearly, ask them to repeat. Never mention that you are an AI "
                "model or a system prompt."
            )
        )

    async def on_enter(self) -> None:
        # greet the user when the session starts
        self.session.generate_reply(
            instructions="Say a short, friendly greeting and ask the user what they'd like to talk about."
        )


_configure_logging()

server = AgentServer()


def prewarm(proc: JobProcess) -> None:
    # load the VAD model once per worker process and share it across sessions
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    ctx.log_context_fields = {"room": ctx.room.name}
    logger.info("starting agent for room %s", ctx.room.name)

    turn_handling = {
        "endpointing": {
            # Deepgram's final transcript can arrive after the first endpoint
            # signal; leave enough time for the final words to be included.
            "min_delay": 0.6,
            "max_delay": 1.2,
        },
        "interruption": {
            # user talks over the agent -> agent stops speaking and listens again
            "enabled": True,
            # Avoid the quota-limited remote adaptive detector; local VAD is
            # predictable and does not silently switch modes mid-session.
            "mode": "vad",
            "min_duration": 0.25,
            "min_words": 2,
            "resume_false_interruption": True,
            "false_interruption_timeout": 0.8,
        },
        # start generating (and optionally speaking) before the user fully finishes
        "preemptive_generation": {"enabled": True, "preemptive_tts": False, "max_retries": 3},
    }
    if TURN_DETECTOR is not None:
        turn_handling["turn_detection"] = TURN_DETECTOR

    stt_kwargs = {
        "model": os.environ.get("DEEPGRAM_MODEL", "nova-3"),
        "language": STT_LANGUAGE,
    }
    if STT_KEYTERMS:
        stt_kwargs["keyterms"] = STT_KEYTERMS

    logger.info(
        "CONFIG llm=%s temp=%s stt=%s/%s tts=sonic-3 voice=%s emotion=%s speed=%s",
        SLOW_LLM_MODEL,
        LLM_TEMPERATURE,
        stt_kwargs["model"],
        STT_LANGUAGE,
        CARTESIA_VOICE[:8] + "...",
        CARTESIA_EMOTION,
        CARTESIA_SPEED,
    )

    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(**stt_kwargs),
        llm=openai.LLM(
            model=SLOW_LLM_MODEL,
            api_key=os.environ["GROQ_API_KEY"],
            base_url=GROQ_BASE_URL,
            temperature=LLM_TEMPERATURE,
        ),
        tts=cartesia.TTS(
            model="sonic-3",
            voice=CARTESIA_VOICE,
            emotion=CARTESIA_EMOTION,
            speed=CARTESIA_SPEED,
        ),
        turn_handling=TurnHandlingOptions(**turn_handling),
        # give the browser a moment to calibrate acoustic echo cancellation so
        # the agent doesn't interrupt itself
        aec_warmup_duration=3.0,
        tts_text_transforms=["filter_emoji", "filter_markdown"],
    )

    tracker = TurnTracker()

    # transcribe events are published back into the room automatically so the
    # frontend can render a live transcript
    @session.on("user_input_transcribed")
    def _on_user_transcript(ev) -> None:
        tracker.on_user_transcript(ev)

    @session.on("user_transcription_timeout")
    def _on_user_timeout(ev) -> None:
        tracker.on_user_timeout(ev)

    @session.on("agent_state_changed")
    def _on_agent_state(ev) -> None:
        tracker.on_agent_state(ev)

    @session.on("conversation_item_added")
    def _on_conversation_item(ev) -> None:
        tracker.on_conversation_item(ev)

    @session.on("speech_created")
    def _on_speech_created(ev) -> None:
        tracker.on_speech_created(ev)

    @session.on("agent_false_interruption")
    def _on_false_interruption(ev) -> None:
        tracker.on_false_interruption(ev)

    @session.on("overlapping_speech")
    def _on_overlapping_speech(ev) -> None:
        tracker.on_overlapping_speech(ev)

    @session.on("session_usage_updated")
    def _on_usage(ev) -> None:
        tracker.on_usage(ev)

    @session.on("error")
    def _on_error(ev) -> None:
        tracker.on_error(ev)

    @session.on("close")
    def _on_close(ev) -> None:
        logger.info("SESSION_CLOSE reason=%s error=%r", ev.reason, ev.error)
        tracker.print_summary("close")

    await session.start(
        agent=VoiceAgent(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                # Filter steady room noise before VAD and Deepgram receive audio.
                noise_cancellation=noise_cancellation.BVC(),
            ),
        ),
    )

    logger.info("agent session started")


if __name__ == "__main__":
    cli.run_app(server)