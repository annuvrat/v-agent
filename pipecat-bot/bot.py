"""Mia on Pipecat — the learning bot.

Same voice stack as the LiveKit agent (Sarvam STT/TTS + Groq LLM + Silero
VAD), rebuilt as an explicit Pipecat frame pipeline so you can learn the
framework against a baseline you already trust.

RUN (from the repo root):
    pipecat-bot\\.venv\\Scripts\\python pipecat-bot\\bot.py

TEST:
    1. Run token_server.py and open http://127.0.0.1:8000/pipecat.html
       -> Connect. The console joins room "lobby" via a participant token.
       (The main index.html targets the LiveKit agent worker and its
       gateway; it has no idea what a Pipecat bot is, so it sits on
       "initializing" — use the Pipecat console instead.)
    2. Run this bot. It joins the SAME room with its own agent token,
       as a second participant. No dispatch rule, no worker pool —
       just two clients meeting in a room.
    3. Talk. Mic audio -> frames -> Sarvam -> Groq -> Sarvam -> speaker.

TEACHING MAP (one concept per LESSON marker below):
    LESSON 1 - Frames: everything flowing through the system is a typed
               Frame. Processors transform frames and forward them.
    LESSON 2 - Pipeline: the ordered list IS the program. Order matters.
    LESSON 3 - Transport: the ears and mouth. ~10 swappable lines; all
               media/network handling lives here, nowhere else.
    LESSON 4 - Services: stateless adapters from frames to provider APIs.
    LESSON 5 - Context/aggregators: how conversation memory is built back
               up out of frames, one side at a time.
    LESSON 6 - Worker/Runner/tracing: who drives the frames, lifecycle,
               and how to SEE a turn as OpenTelemetry spans.
    LESSON 7 - Custom processors: TranscriptTap observes frames and forwards
               text to the browser over LiveKit data without disturbing flow.
    DENOISE  - audio_in_filter=RNNoiseFilter() on the transport: the Pipecat
               equivalent of agent.py's noise_cancellation.BVC().
"""

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

# Config first: reuse the repo-root .env so BOTH frameworks share one set of
# keys (LIVEKIT_*, GROQ_*, SARVAM_*). Nothing Pipecat-specific to configure.
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# The bot needs a room name; the root .env doesn't define one, so default to
# the same room the browser page joins. Override with LIVEKIT_ROOM_NAME=...
os.environ.setdefault("LIVEKIT_ROOM_NAME", "lobby")

from loguru import logger  # same logger Pipecat itself uses (stdlib logging
# has no handler configured in this process, so logging.getLogger() lines
# would silently vanish — lesson: match the framework's logging)

from pipecat.audio.filters.rnnoise_filter import RNNoiseFilter
from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    Frame,
    InterimTranscriptionFrame,
    LLMFullResponseEndFrame,
    LLMTextFrame,
    TTSSpeakFrame,
    TTSTextFrame,
    TranscriptionFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.runner.livekit import configure
from pipecat.services.groq.llm import GroqLLMService
from pipecat.services.sarvam.stt import SarvamSTTService
from pipecat.services.sarvam.tts import SarvamTTSService
from pipecat.transports.livekit.transport import (
    LiveKitOutputTransportMessageFrame,
    LiveKitParams,
    LiveKitTransport,
)
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.utils.tracing.setup import setup_tracing
from pipecat.workers.runner import WorkerRunner

logger = logger.bind(bot="mia-pipecat")

SYSTEM_INSTRUCTION = (
    "You are Mia, a friendly voice assistant. Your replies will be spoken "
    "aloud, so keep them to one or two short sentences. No emojis, no bullet "
    "points, no markdown. If the user speaks Hindi or Hinglish, reply in the "
    "same language."
)


def _require(name: str) -> str:
    """Fail fast with a readable error instead of a cryptic 401 later."""
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} is not set (check the repo-root .env)")
    return value


class SafeRNNoiseFilter(RNNoiseFilter):
    """Hardened RNNoise: a filter must never be able to kill the audio path.

    Two real failure modes found by testing:
      1. Odd-byte chunks raise ValueError inside pyrnnoise's frombuffer
         (int16 needs even byte counts). The framework's audio task only
         catches TimeoutError, so ONE such chunk ends audio FOREVER.
      2. Any other filter exception would do the same.
    This subclass pads odd chunks and passes audio through untouched on
    any failure — worst case is unfiltered audio, never silence. It also
    logs flow stats so the next debug run shows whether audio is even
    reaching the filter (input bytes + RMS every ~10s of audio).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._calls = 0

    async def filter(self, audio: bytes) -> bytes:
        try:
            if len(audio) % 2:
                audio += b"\x00"
            out = await super().filter(audio)
            self._calls += 1
            if self._calls == 1 or self._calls % 500 == 0:
                import numpy as np

                rms = 0.0
                if len(out) >= 2:
                    s = np.frombuffer(out, dtype=np.int16).astype(np.float32)
                    rms = float(np.sqrt((s**2).mean()))
                logger.info(
                    f"noise filter flow: calls={self._calls} "
                    f"in_bytes={len(audio)} out_bytes={len(out)} out_rms={rms:.0f}"
                )
            return out
        except Exception as e:
            logger.warning(f"noise filter failed, passing audio through: {e}")
            return audio


class TranscriptTap(FrameProcessor):
    """LESSON 7 (your first custom processor): forward transcript text to the
    browser console over LiveKit data messages.

    A processor is just a stage that sees every frame passing through its
    position in the pipeline. This one forwards text and passes EVERYTHING
    downstream untouched — it observes without interfering.

    Two instances, two viewpoints:
      - after `stt`  (speaker="user"):  sees TranscriptionFrames as Sarvam
        streams them — interim updates plus the sealed final.
      - after `llm`  (speaker="agent"): accumulates the reply's text frames
        and flushes one clean bubble on LLMFullResponseEndFrame. It also
        catches TTSSpeakFrame (the canned greeting, which bypasses the LLM)
        — but only labels it when this tap is the agent-side one.
    """

    def __init__(self, transport: LiveKitTransport, speaker: str):
        super().__init__()
        self._transport = transport
        self._speaker = speaker
        self._text = ""

    async def _send(self, text: str, final: bool) -> None:
        if not text or not text.strip():
            return
        # The output transport owns the room connection: handing it a
        # message frame publishes the dict as JSON over a reliable LiveKit
        # data message. (There is no transport.send_data() — that method
        # lives on the internal client object. Calling it on the facade
        # raised AttributeError and swallowed the frame: the bug that had
        # muted Mia.) pipecat.html filters on sender identity + this JSON
        # shape (data messages carry no topic, so the envelope IS routing).
        frame = LiveKitOutputTransportMessageFrame(
            message={
                "type": "transcript",
                "speaker": self._speaker,
                "text": text,
                "final": final,
            }
        )
        await self._transport.output().send_message(frame)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        # A tap must NEVER break the pipeline: if sending fails, log and
        # keep the frames flowing. push_frame always runs (outside the try).
        try:
            if isinstance(frame, TranscriptionFrame):
                await self._send(frame.text, frame.finalized)
            elif isinstance(frame, InterimTranscriptionFrame):
                await self._send(frame.text, False)
            elif isinstance(frame, (LLMTextFrame, TTSTextFrame)):
                self._text += frame.text
            elif isinstance(frame, TTSSpeakFrame) and self._speaker == "agent":
                await self._send(frame.text, True)
            elif isinstance(frame, LLMFullResponseEndFrame):
                text, self._text = self._text, ""
                await self._send(text, True)
        except Exception as e:
            logger.warning(f"TranscriptTap({self._speaker}): transcript send failed: {e}")
        await self.push_frame(frame, direction)


async def main() -> None:
    # LESSON 6 (tracing): one env flag turns frames into console spans.
    # Watch a full turn — VAD -> STT -> LLM -> TTS — print as nested spans.
    # Set ENABLE_TRACING=0 to silence it.
    if os.getenv("ENABLE_TRACING", "1") == "1":
        setup_tracing(service_name="mia-pipecat", console_export=True)

    # LESSON 3 (transport, part 1): configure() mints an AGENT token with
    # generate_token_with_agent() — the JWT that tells LiveKit's SFU "this
    # client is a bot, allowed to publish audio". Compare with token_server.py,
    # which mints PARTICIPANT tokens for browsers. Same room, two roles.
    (url, token, room_name) = await configure()
    logger.info(f"joining LiveKit room {room_name!r}")

    # Noise filtering lives HERE, not in the pipeline: the transport runs
    # every inbound audio frame through audio_in_filter BEFORE VAD, STT, or
    # smart-turn ever see it. This is the Pipecat equivalent of agent.py's
    # noise_cancellation.BVC() on audio input — denoise once at the edge and
    # every downstream stage benefits (fewer phantom VAD starts, cleaner
    # transcripts, no more noise-flipped "listening" states).
    # RNNoise is a small recurrent net, strong on steady noise (fans, hum,
    # room tone); it needs no account and no key. Krisp-grade models
    # (KrispVivaFilter) suppress background voices too, but need an API key
    # plus a .kef model file — the upgrade path if voices leak through.
    # Toggle live with ENABLE_NOISE_FILTER=0.
    # NOTE (2026-10-08 incident): RNNoise is default-OFF until the fidelity
    # question is resolved. Evidence: the pre-filter bot transcribed normal
    # speech fine; with SafeRNNoiseFilter in path, ~25s of user speech
    # produced only empty STT results plus one loud isolated "Hello." —
    # while VAD kept firing on energy. Prime suspects: the 16k->48k->16k
    # double resample (QQ quality) mangling phonetics, or pyrnnoise
    # suppressing conversational-level speech. Re-enable to investigate
    # with ENABLE_NOISE_FILTER=1 (flow logs show in/out bytes + RMS).
    noise_filter = None
    if os.getenv("ENABLE_NOISE_FILTER", "0") == "1":
        noise_filter = SafeRNNoiseFilter()
        logger.info("noise filter: RNNoise enabled on audio input")
    else:
        logger.info("noise filter: disabled")

    # LESSON 3 (transport, part 2): the ONLY media code in the whole file.
    # Swap these ~10 lines for DailyTransport/FastAPIWebsocketTransport and
    # everything below is byte-identical. Video stays off: audio-only bot.
    transport = LiveKitTransport(
        url=url,
        token=token,
        room_name=room_name,
        params=LiveKitParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_filter=noise_filter,
        ),
    )

    # LESSON 4 (services): thin adapters — frames in, provider API out.
    # Same providers as agent.py, same env knobs, same values.
    stt = SarvamSTTService(
        api_key=_require("SARVAM_API_KEY"),
        mode=os.getenv("SARVAM_STT_MODE", "transcribe"),
        settings=SarvamSTTService.Settings(
            model=os.getenv("SARVAM_STT_MODEL", "saaras:v4"),
            language=os.getenv("SARVAM_LANGUAGE", "en-IN"),
            high_vad_sensitivity=True,
        ),
    )

    llm = GroqLLMService(
        api_key=_require("GROQ_API_KEY"),
        settings=GroqLLMService.Settings(
            model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
            temperature=0.7,
            system_instruction=SYSTEM_INSTRUCTION,
        ),
    )

    tts = SarvamTTSService(
        api_key=_require("SARVAM_API_KEY"),
        settings=SarvamTTSService.Settings(
            model=os.getenv("SARVAM_TTS_MODEL", "bulbul:v3"),
            voice=os.getenv("SARVAM_TTS_VOICE") or None,
            language=os.getenv("SARVAM_LANGUAGE", "en-IN"),
            pace=float(os.getenv("SARVAM_TTS_PACE", "1.0")),
        ),
    )

    # LESSON 5 (memory): the LLM is stateless, so history must be rebuilt
    # from frames. The user aggregator appends what YOU said; the assistant
    # aggregator — sitting AFTER transport.output() so only actually-spoken
    # audio is recorded — appends what the BOT said. Together they maintain
    # one LLMContext across turns.
    # LESSON 2b (endpointing = latency): the DEFAULT stop strategy is
    # TurnAnalyzer + LocalSmartTurn with stop_secs=3 — the model may wait up
    # to THREE silent seconds before calling a turn complete. That was the
    # "Silence in ms: 3025" tail in the logs: the single biggest latency term
    # (Sarvam STT ~0.1s, Groq ~0.7s, Sarvam TTS ~0.3s are all smaller).
    # We keep smart-turn's completeness judgments but shorten its tail.
    # Tune live with SMART_TURN_STOP_SECS=0.8 (seconds of trailing silence).
    # One-line alternative, LiveKit-style fixed endpointing with no ML
    # judgment (deterministic VAD + timer turns):
    #   stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=0.6)]
    # from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy
    smart_turn_stop_secs = float(os.getenv("SMART_TURN_STOP_SECS", "0.8"))
    logger.info(f"endpointing smart_turn_stop_secs={smart_turn_stop_secs}")
    user_strategies = UserTurnStrategies(
        stop=[
            TurnAnalyzerUserTurnStopStrategy(
                turn_analyzer=LocalSmartTurnAnalyzerV3(
                    params=SmartTurnParams(stop_secs=smart_turn_stop_secs)
                )
            )
        ]
    )

    context = LLMContext()
    user_agg, assistant_agg = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(),
            user_turn_strategies=user_strategies,
        ),
    )

    # LESSON 1+2 (frames + pipeline): this ordered list IS the program.
    # Audio flows down: input -> STT -> user memory -> LLM -> TTS -> output
    # -> assistant memory. Each stage emits typed frames the next stage
    # consumes (AudioRawFrame -> TranscriptionFrame -> LLMTextFrame ->
    # TTSAudioRawFrame). To add a stage (filter, tool, recorder), you insert
    # a processor here — nothing else changes.
    pipeline = Pipeline(
        [
            transport.input(),  # mic audio in: AudioRawFrame stream
            stt,  # audio -> text: TranscriptionFrame
            TranscriptTap(transport, "user"),  # LESSON 7: forward user text
            user_agg,  # text -> memory: appends to LLMContext
            llm,  # memory -> reply: LLMTextFrame stream
            TranscriptTap(transport, "agent"),  # LESSON 7: forward reply text
            tts,  # reply -> speech: TTSAudioRawFrame stream
            transport.output(),  # speech -> speaker
            assistant_agg,  # spoken reply -> memory
        ]
    )

    # LESSON 6 (lifecycle): the worker OWNS the pipeline — it pumps frames,
    # tracks turns, and (with enable_tracing) emits spans. The runner just
    # hosts workers and blocks until done. idle_timeout_secs=None parks the
    # bot in the room instead of self-cancelling after 5 idle minutes.
    worker = PipelineWorker(
        pipeline,
        name="mia-pipecat",
        params=PipelineParams(
            audio_out_sample_rate=24000,  # matches Sarvam bulbul default
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        enable_tracing=os.getenv("ENABLE_TRACING", "1") == "1",
        enable_turn_tracking=True,
        idle_timeout_secs=None,
    )

    # Event hooks: the transport tells us about room life; we inject frames.
    # A TTSSpeakFrame here speaks WITHOUT waking the LLM — a greeting is
    # canned speech, not a conversation turn.
    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport, participant_id):
        logger.info(f"participant {participant_id} joined; greeting")
        await worker.queue_frame(
            TTSSpeakFrame(
                "Hey! I'm Mia running on Pipecat now. Same voice, same brain, "
                "different engine. What's on your mind?"
            )
        )

    # Deliberately NOT cancelling here: this bot is parked in a lobby-style
    # room, so a browser refresh shouldn't kill it. (For 1:1 call bots you'd
    # call await worker.cancel() instead — compare the voice examples.)
    @transport.event_handler("on_participant_disconnected")
    async def on_participant_disconnected(transport, participant_id):
        logger.info(f"participant {participant_id} left; staying parked")

    runner = WorkerRunner()
    await runner.add_workers(worker)
    await runner.run()


if __name__ == "__main__":
    asyncio.run(main())
