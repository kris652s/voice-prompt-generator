 from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os, tempfile, time, logging
from collections import deque
from openai import OpenAI

app = Flask(__name__, static_folder="static")

# -------- CORS + upload limits --------
ALLOWED_ORIGIN = os.getenv("ALLOWED_ORIGIN", "*")
CORS(app, resources={r"/*": {"origins": ALLOWED_ORIGIN}})
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25MB

# -------- Logging --------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("voice-prompt")

# -------- Simple per-IP rate limiter --------
RATE_WINDOW_SEC = 60         # 1 minute window
RATE_MAX_REQUESTS = 6        # 6 requests / minute / IP
_ip_hits: dict[str, deque] = {}

def _too_many(ip: str) -> bool:
    now = time.time()
    q = _ip_hits.setdefault(ip, deque())
    # drop old
    while q and q[0] < now - RATE_WINDOW_SEC:
        q.popleft()
    if len(q) >= RATE_MAX_REQUESTS:
        return True
    q.append(now)
    return False

# -------- Lazy OpenAI client --------
_client = None
def get_client():
    global _client
    if _client is None:
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        _client = OpenAI(api_key=key)
    return _client

# -------- Health & static --------
@app.get("/health")
def health():
    return {"ok": True}

@app.get("/")
def root():
    return send_from_directory(app.static_folder, "index.html")

# -------- Main endpoint --------
@app.post("/process-voice")
def process_voice():
    ip = (request.headers.get("x-forwarded-for") or request.remote_addr or "").split(",")[0].strip()
    if _too_many(ip):
        return jsonify({"error": "Too many requests. Try again in a minute."}), 429

    if "audio" not in request.files:
        return jsonify({"error": "No 'audio' file in form-data"}), 400

    model = (request.form.get("model") or "gpt-4o-mini").strip()
    up = request.files["audio"]

    # Save upload to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
        up.save(tmp.name)
        path = tmp.name

    try:
        client = get_client()
        log.info("IP %s - processing audio (%s bytes) model=%s", ip, up.content_length, model)

        # 1) Speech → English: try translations first
        try:
            transcript_text = client.audio.translations.create(
                model="whisper-1",
                file=open(path, "rb"),
                response_format="text",
                temperature=0
            ).strip()
        except Exception as e1:
            log.warning("translations failed (%s), falling back to transcription", e1)
            raw_text = client.audio.transcriptions.create(
                model="whisper-1",
                file=open(path, "rb"),
                response_format="text",
                temperature=0
            ).strip()
            tr = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Translate to natural English while preserving meaning."},
                    {"role": "user", "content": raw_text}
                ],
                temperature=0
            )
            transcript_text = tr.choices[0].message.content.strip()

        # 2) Refine into a clean, actionable prompt (use a cheap fast model)
        refine = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content":
                 "You are a prompt engineer. Given the user's raw intent, output a single, clear, action-oriented prompt an LLM can execute directly. Fix grammar, add obvious specifics, keep it concise. Output ONLY the final prompt."},
                {"role": "user", "content": transcript_text}
            ],
            temperature=0
        )
        refined_prompt = refine.choices[0].message.content.strip()

        # 3) Execute the refined prompt (use selected model)
        final = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": refined_prompt}],
            temperature=0.2
        )
        final_text = final.choices[0].message.content.strip()

        return jsonify({"refined_prompt": refined_prompt, "response": final_text})

    except Exception as e:
        log.exception("process_voice failed")
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            os.remove(path)
        except Exception:
            pass

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
