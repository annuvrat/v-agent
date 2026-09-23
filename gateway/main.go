package main

import (
	"encoding/json"
	"log"
	"net/http"
	"os"
	"time"

	"github.com/gorilla/websocket"
)

const (
	writeWait      = 10 * time.Second
	heartbeatEvery = 20 * time.Second
	maxMessageSize = 64 * 1024
	queueSize      = 32
)

var upgrader = websocket.Upgrader{
	ReadBufferSize:  4 * 1024,
	WriteBufferSize: 4 * 1024,
	CheckOrigin: func(r *http.Request) bool {
		return true
	},
}
var gatewaySecret string

type envelope struct {
	Type      string `json:"type"`
	SessionID string `json:"session_id,omitempty"`
	Room      string `json:"room,omitempty"`
	Token     string `json:"token,omitempty"`
	Message   string `json:"message,omitempty"`
	At        string `json:"at,omitempty"`
}

func main() {
	gatewaySecret = os.Getenv("GATEWAY_SECRET")
	if gatewaySecret == "" {
		log.Fatal("GATEWAY_SECRET is required")
	}
	port := os.Getenv("GATEWAY_PORT")
	if port == "" {
		port = "8081"
	}

	http.HandleFunc("/health", healthHandler)
	http.HandleFunc("/ws", websocketHandler)

	log.Printf("voice gateway listening on http://127.0.0.1:%s", port)
	log.Fatal(http.ListenAndServe(":"+port, nil))
}

func healthHandler(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	_, _ = w.Write([]byte(`{"ok":true,"service":"voice-gateway"}`))
}

func websocketHandler(w http.ResponseWriter, r *http.Request) {
	connection, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Printf("websocket upgrade failed: %v", err)
		return
	}
	defer connection.Close()

	sessionID := newSessionID()
	outbound := make(chan []byte, queueSize)
	done := make(chan struct{})

	go writeLoop(connection, outbound, done)
	if !enqueue(outbound, envelope{
		Type:      "session.ready",
		SessionID: sessionID,
		Message:   "gateway connection established",
		At:        time.Now().UTC().Format(time.RFC3339Nano),
	}) {
		return
	}

	connection.SetReadLimit(maxMessageSize)
	_ = connection.SetReadDeadline(time.Now().Add(heartbeatEvery * 2))
	connection.SetPongHandler(func(string) error {
		return connection.SetReadDeadline(time.Now().Add(heartbeatEvery * 2))
	})

	for {
		messageType, payload, err := connection.ReadMessage()
		if err != nil {
			log.Printf("session %s closed: %v", sessionID, err)
			close(done)
			return
		}
		if messageType != websocket.TextMessage {
			continue
		}

		var incoming envelope
		if err := json.Unmarshal(payload, &incoming); err != nil {
			if !enqueue(outbound, envelope{Type: "error", Message: "invalid JSON message"}) {
				return
			}
			continue
		}
		if incoming.Type == "session.start" {
			if incoming.Token != gatewaySecret {
				if !enqueue(outbound, envelope{
					Type:    "error",
					Message: "unauthorized",
				}) {
					return
				}
				continue
			}

			if incoming.Room == "" {
				if !enqueue(outbound, envelope{Type: "error", Message: "room is required"}) {
					return
				}
				continue
			}
			if !enqueue(outbound, envelope{
				Type:      "session.accepted",
				SessionID: sessionID,
				Room:      incoming.Room,
				Message:   "session accepted",
				At:        time.Now().UTC().Format(time.RFC3339Nano),
			}) {
				return
			}
			continue
		}
		if incoming.Type == "session.ping" {
			if !enqueue(outbound, envelope{Type: "session.pong", SessionID: sessionID, At: time.Now().UTC().Format(time.RFC3339Nano)}) {
				return
			}
		}
	}
}

func writeLoop(connection *websocket.Conn, outbound <-chan []byte, done <-chan struct{}) {
	heartbeat := time.NewTicker(heartbeatEvery)
	defer heartbeat.Stop()

	for {
		select {
		case payload := <-outbound:
			_ = connection.SetWriteDeadline(time.Now().Add(writeWait))
			if err := connection.WriteMessage(websocket.TextMessage, payload); err != nil {
				return
			}
		case <-heartbeat.C:
			_ = connection.SetWriteDeadline(time.Now().Add(writeWait))
			if err := connection.WriteMessage(websocket.PingMessage, nil); err != nil {
				return
			}
		case <-done:
			return
		}
	}
}

func enqueue(outbound chan<- []byte, message envelope) bool {
	payload, err := json.Marshal(message)
	if err != nil {
		return false
	}
	select {
	case outbound <- payload:
		return true
	default:
		return false
	}
}

func newSessionID() string {
	return time.Now().UTC().Format("20060102T150405.000000000Z07:00")
}
