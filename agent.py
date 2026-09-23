import logging
import os

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
from livekit.plugins import cartesia, deepgram, openai, silero

logger = logging.getLogger("voice-agent")

load_dotenv()

SLOW_LLM_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Use LiveKit's built-in semantic turn detector ("end of turn" model).
# It ships with livekit-agents (>=1.6.1); if unavailable on your version we
# simply fall back to the session's automatic turn detection.
try:
    from livekit.agents.inference import TurnDetector

    TURN_DETECTOR = TurnDetector()
except ImportError:  # pragma: no cover
    TURN_DETECTOR = None


class VoiceAgent(Agent):
    """The agent's personality / system prompt lives here."""

    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are Mia, a friendly and curious voice assistant built with LiveKit. "
                "You speak to the user over voice in a casual, warm tone. "
                "Keep your responses concise and conversational - a sentence or two is "
                "usually enough unless the user asks for detail. "
                "Never use emojis, markdown, aal charsterisks, or speciacters; just plain "
                "spoken text. Do not mention that you are an AI model or a system prompt. "
                "If you did not hear the user clearly, ask them to repeat."
            )
        )

    async def on_enter(self) -> None:
        # greet the user when the session starts
        self.session.generate_reply(
            instructions="Say a short, friendly greeting and ask the user what they'd like to talk about."
        )


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
            "min_duration": 0.15,
            "min_words": 0,
            "resume_false_interruption": True,
            "false_interruption_timeout": 0.8,
        },
        # start generating (and optionally speaking) before the user fully finishes
        "preemptive_generation": {"enabled": True, "preemptive_tts": False, "max_retries": 3},
    }
    if TURN_DETECTOR is not None:
        turn_handling["turn_detection"] = TURN_DETECTOR

    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(
            model=os.environ.get("DEEPGRAM_MODEL", "nova-3"),
        ),
        llm=openai.LLM(
            model=SLOW_LLM_MODEL,
            api_key=os.environ["GROQ_API_KEY"],
            base_url=GROQ_BASE_URL,
            temperature=0.4,
        ),
        tts=cartesia.TTS(
            model="sonic-3",
            voice=os.environ.get("CARTESIA_VOICE", "9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"),
        ),
        turn_handling=TurnHandlingOptions(**turn_handling),
        # give the browser a moment to calibrate acoustic echo cancellation so
        # the agent doesn't interrupt itself
        aec_warmup_duration=3.0,
        tts_text_transforms=["filter_emoji", "filter_markdown"],
    )

    # transcribe events are published back into the room automatically so the
    # frontend can render a live transcript
    @session.on("user_input_transcribed")
    def _on_user_transcript(ev) -> None:
        logger.debug("user said: %s", ev.transcript)

    @session.on("agent_speech_created")
    def _on_agent_speech(ev) -> None:
        logger.debug("agent said: %s", ev.utterance_text)

    await session.start(
        agent=VoiceAgent(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                # uncomment to enable noise cancellation: python -m pip install livekit-plugins-noise-cancellation
                # noise_cancellation=...,
            ),
        ),
    )

    logger.info("agent session started")
    

if __name__ == "__main__":
    cli.run_app(server)