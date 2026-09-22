import os
import uuid
from datetime import timedelta
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from livekit.api import AccessToken, VideoGrants

load_dotenv()

LIVEKIT_URL = os.getenv("LIVEKIT_URL")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")

BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"

app = FastAPI(title="Realtime Voice Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/get_token")
async def get_token(
    room: str = Query(default="lobby", description="Room name to join"),
    identity: str = Query(default=None, description="Optional participant identity"),
) -> JSONResponse:
    """Issue a short-lived LiveKit access token so the browser can join a room."""
    if not identity:
        identity = f"user-{uuid.uuid4().hex[:8]}"
    if not room:
        raise ValueError("room must be a non-empty string")

    token = (
        AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
        .with_identity(identity)
        .with_ttl(timedelta(hours=1))
        .with_grants(
            VideoGrants(
                room=room,
                room_join=True,
                can_publish=True,
                can_subscribe=True,
            )
        )
        .to_jwt()
    )

    return JSONResponse(
        {
            "url": LIVEKIT_URL,
            "token": token,
            "identity": identity,
            "room": room,
        }
    )


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"ok": True})


@app.middleware("http")
async def no_store_frontend(request, call_next):
    """Never let the browser keep a stale cached copy of the (zero-build)
    frontend — a 304 of an older index.html was the cause of the
    "Connect does nothing / stays on connecting" symptom."""
    response = await call_next(request)
    if request.url.path in ("/", "/index.html"):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return response


# Serve the single-page frontend last so API routes take precedence.
app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
