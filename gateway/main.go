package main

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"log"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/gorilla/websocket"
	"github.com/joho/godotenv"
	"github.com/livekit/protocol/auth"
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
	_ = godotenv.Load(".env")
	_ = godotenv.Load("../.env")

	gatewaySecret = os.Getenv("GATEWAY_SECRET")
	if gatewaySecret == "" {
		gatewaySecret = "dev-only-change-me"
	}

	port := os.Getenv("PORT")
	if port == "" {
		port = os.Getenv("GATEWAY_PORT")
	}
	if port == "" {
		port = "8000"
	}

	webDir := os.Getenv("WEB_DIR")
	if webDir == "" {
		if _, err := os.Stat("web"); err == nil {
			webDir = "web"
		} else if _, err := os.Stat("../web"); err == nil {
			webDir = "../web"
		} else {
			webDir = "web"
		}
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/health", healthHandler)
	mux.HandleFunc("/get_token", getTokenHandler)
	mux.HandleFunc("/get_gateway_token", getGatewayTokenHandler)
	mux.HandleFunc("/ws", websocketHandler)
	mux.Handle("/", staticFileServer(webDir))

	log.Printf("server listening on http://0.0.0.0:%s (serving web from %s)", port, webDir)
	log.Fatal(http.ListenAndServe(":"+port, corsMiddleware(mux)))
}

func corsMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "*")
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusOK)
			return
		}
		next.ServeHTTP(w, r)
	})
}

func healthHandler(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	_, _ = w.Write([]byte(`{"ok":true,"service":"realtime-voice-gateway"}`))
}

var (
	rateLimitMu sync.Mutex
	ipRequests  = make(map[string][]time.Time)
)

func checkRateLimit(ip string, limit int, window time.Duration) bool {
	rateLimitMu.Lock()
	defer rateLimitMu.Unlock()

	now := time.Now()
	cutoff := now.Add(-window)

	var valid []time.Time
	for _, t := range ipRequests[ip] {
		if t.After(cutoff) {
			valid = append(valid, t)
		}
	}

	if len(valid) >= limit {
		ipRequests[ip] = valid
		return false
	}

	ipRequests[ip] = append(valid, now)
	return true
}

func getClientIP(r *http.Request) string {
	if xff := r.Header.Get("X-Forwarded-For"); xff != "" {
		parts := strings.Split(xff, ",")
		return strings.TrimSpace(parts[0])
	}
	if xrip := r.Header.Get("X-Real-IP"); xrip != "" {
		return strings.TrimSpace(xrip)
	}
	return r.RemoteAddr
}

func getTokenHandler(w http.ResponseWriter, r *http.Request) {
	clientIP := getClientIP(r)
	if !checkRateLimit(clientIP, 25, time.Hour) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusTooManyRequests)
		_ = json.NewEncoder(w).Encode(map[string]string{
			"error": "Demo rate limit reached (25 calls/hour). Please contact Annuvrat to schedule a dedicated demo!",
		})
		return
	}

	visitorName := strings.TrimSpace(r.URL.Query().Get("visitor_name"))
	if visitorName == "" {
		visitorName = "Guest Visitor"
	}

	log.Printf("===> [RECRUITER / VISITOR] Name: %q | IP: %s | Time: %s", visitorName, clientIP, time.Now().Format("2006-01-02 15:04:05"))

	room := r.URL.Query().Get("room")
	if room == "" {
		room = "lobby"
	}
	identity := r.URL.Query().Get("identity")
	if identity == "" {
		b := make([]byte, 4)
		_, _ = rand.Read(b)
		identity = "user-" + hex.EncodeToString(b)
	}

	apiKey := os.Getenv("LIVEKIT_API_KEY")
	apiSecret := os.Getenv("LIVEKIT_API_SECRET")
	livekitURL := os.Getenv("LIVEKIT_URL")

	if apiKey == "" || apiSecret == "" {
		http.Error(w, `{"error":"LIVEKIT_API_KEY or LIVEKIT_API_SECRET not configured"}`, http.StatusInternalServerError)
		return
	}

	at := auth.NewAccessToken(apiKey, apiSecret)
	grant := &auth.VideoGrant{
		Room:     room,
		RoomJoin: true,
	}
	grant.SetCanPublish(true)
	grant.SetCanSubscribe(true)

	at.AddGrant(grant).
		SetIdentity(identity).
		SetName(visitorName).
		SetValidFor(15 * time.Minute)

	token, err := at.ToJWT()
	if err != nil {
		log.Printf("failed to generate token: %v", err)
		http.Error(w, `{"error":"failed to generate LiveKit access token"}`, http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]string{
		"url":          livekitURL,
		"token":        token,
		"identity":     identity,
		"visitor_name": visitorName,
		"room":         room,
	})
}

func getGatewayTokenHandler(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]string{
		"gateway_token": gatewaySecret,
	})
}

func staticFileServer(webDir string) http.Handler {
	fs := http.FileServer(http.Dir(webDir))
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/" || r.URL.Path == "/index.html" {
			w.Header().Set("Cache-Control", "no-cache, no-store, must-revalidate")
		}
		fs.ServeHTTP(w, r)
	})
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
