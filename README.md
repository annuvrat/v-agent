# Realtime Voice Agent

A browser voice assistant built on [LiveKit Agents](https://docs.livekit.io/agents/)
— Deepgram STT → Groq LLM → Cartesia TTS — with a FastAPI token server and a
zero-build vanilla-JS frontend. Talk to it from any WebRTC-capable browser;
no app install required.

## Architecture

```
Browser (web/index.html, livekit-client)
   │  WebRTC audio/video into a LiveKit room
   ▼
LiveKit Cloud
   ▲
   │  livekit-agents worker (agent.py)
   └── Deepgram STT (nova-3) → Groq LLM (llama-3.3) → Cartesia TTS (sonic-3)
       Silero VAD · semantic turn detection via TurnDetector
                         ▲
        token_server.py (FastAPI) issues JWT via /get_token, serves web/

   Optional control plane:
   Browser control events -> gateway/ (Go WebSocket gateway) -> future gRPC session bridge
```

## Prerequisites

- Python 3.11+ (project uses `.venv`).
- A LiveKit Cloud project: URL + API key + secret (free tier works).
- API keys: Deepgram, Groq, Cartesia.

## Setup

1. Create the virtualenv and install dependencies
   (the bundled mirror is used if pypi is blocked on your network):

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt `
     -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
   ```

2. Configure credentials:

   ```powershell
   Copy-Item .env.example .env
   # then fill in LIVEKIT_URL / LIVEKIT_API_KEY / LIVEKIT_API_SECRET
   #        DEEPGRAM_API_KEY / GROQ_API_KEY / CARTESIA_API_KEY
   ```

3. Install SpeechKit model files (TurnDetector + Silero VAD) once:

   ```powershell
   .\.venv\Scripts\python.exe -m livekit.agents download-files
   ```

## Run

Open **two** terminals.

Terminal 1 — the token server (also serves the frontend):

```powershell
.\.venv\Scripts\python.exe token_server.py
```

Terminal 2 — the agent worker:

```powershell
.\.venv\Scripts\python.exe agent.py dev
```

Then open <http://127.0.0.1:8000>, click **Connect**, and start talking.

## Files

| File                    | Purpose                                                                |
| ----------------------- | ---------------------------------------------------------------------- |
| `agent.py`              | LiveKit Agents worker: session wiring, turn handling, plugins          |
| `token_server.py`       | FastAPI app: LiveKit JWT endpoint + static frontend host               |
| `web/index.html`        | Single-page browser client (livekit-client via CDN)                    |
| `gateway/`              | Optional Go WebSocket control gateway; LiveKit remains the media plane |
| `requirements.txt`      | Dependencies                                                           |
| `.env.example` / `.env` | Credentials                                                            |

## Troubleshooting

- **pypi unreachable**: use the Tsinghua mirror (`-i` above); it is reliable
  from this machine.
- **TurnDetector model missing**: run `download-files`; without it, turn
  detection may fall back to local processing.
- **Agent says it can't hear you**: confirm your browser's mic permission and
  that you are joining the same room in `web/index.html` and `get_token`.

## Run the optional Go gateway

Install Go 1.22 or newer, then start it from the repository root:

```powershell
cd gateway
go mod tidy
go run .
```

The gateway listens on `http://127.0.0.1:8081`. Check `GET /health` or connect
to `/ws` with a WebSocket client. It currently handles session lifecycle and
heartbeats only. Keep voice media on LiveKit WebRTC; add a gRPC bridge after
the control protocol is stable.
