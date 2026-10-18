# syntax=docker/dockerfile:1

# Stage 1: Compile the lightweight Go token & web server
FROM golang:alpine AS builder
WORKDIR /build

RUN apk add --no-cache git ca-certificates

COPY gateway/go.mod gateway/go.sum ./
RUN go mod download

COPY gateway/ ./
RUN CGO_ENABLED=0 GOOS=linux go build -ldflags="-s -w" -o server .

# Stage 2: Python runtime with LiveKit Agent
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies for WebRTC audio/codecs
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    ffmpeg \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download TurnDetector & Silero VAD models at image build time
RUN python -m livekit.agents download-files

# Copy compiled Go server from builder stage
COPY --from=builder /build/server /app/server

# Copy application files
COPY . /app

# Ensure executable permissions for runner script
RUN chmod +x /app/start.sh

ENV PORT=8000
EXPOSE 8000

CMD ["/app/start.sh"]
