 import os
import time
import logging
from collections import defaultdict, deque

from flask import (
    Flask,
    request,
    jsonify,
    send_from_directory,
    Response,
    stream_with_context,
)
from flask_cors import CORS
from openai import OpenAI
import tempfile

# ----------------------------
# App / Config
# ----------------------------
app = Flask(__name__, static_folder="static")

ALLOWED_ORIGIN = os.getenv("ALLOWED_ORIGIN", "*")
CORS(
    app,
    resources={r"/*": {"origins": ALLOWED_ORIGIN}},
    expose_headers=["X-Refined-Prompt", "X-Transcript"]  # allow frontend to read these
)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25 MB

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("voice-prompt")

# Lazy OpenAI client
_client = None
def get_client():
    global _client
    if _client is None:
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        _client = OpenAI(api_key=key)
    return _client

REFINER_MODEL = os.getenv("REFINER_MODEL", "gpt-4o-mini")

# Light per-IP rate limit
RATE_WINDOW_SEC = 60
RATE_MAX_REQUESTS = 6
_ip_hits: dict[str, deque[float]] = defaultdict(deque)


def _too_many(ip: str) -> bool:
    now = time.time()
    dq = _ip_hits[ip]
    dq.append(now)
    while dq and now - dq[0] > RATE_WINDOW_SEC:
        dq.popleft()
    return len(dq) > RATE_MAX_REQUESTS


# ----------------------------
# Utilities
# ----------------------------
def _save_upload_to_temp(audio_file, mime_from_form: str) -> str:
    """
    Saves the uploaded audio to a temp file with a correct extension.
    Supports WebM/Opus (Chrome/Edge/Firefox) and MP4/AAC (Safari/iOS).
    Returns the temp file path.
    """
    mime = (mime_from_form or audio_file.mimetype or "").lower()
    ext = ".mp4" if mime.startswith("audio/mp4") else ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        audio_file.save(tmp.name)
        return tmp.name


def _transcribe_from_path(path: str) -> str:
    """
    Whisper (server-side). Returns the transcribed text.
    """
    client = get_client()
    with open(path, "rb") as f:
        tr = client.audio.transcriptions.create(model="whisper-1", file=f)
    text = (tr.text or "").strip()
    return text


def _refine(raw_text: str) -> str:
    """
    Safe prompt refiner via Chat Completions:
    - Preserve user's meaning.
    - If not English, translate to natural English.
    - Remove filler words; keep essential details.
    - Output ONLY the refined prompt (no preamble).
    """
    raw = (raw_text or "").strip()
    if not raw:
        return raw

    system_msg = (
        "You are a prompt refiner. Rewrite the user's utterance into a clear, concise prompt "
        "for an LLM. Requirements:\n"
        "- Preserve the user's original meaning exactly.\n"
        "- If the text is not in English, translate it to natural English.\n"
        "- Remove filler words and hesitations; keep essential details.\n"
        "- Do NOT invent details or add assumptions.\n"
        "- Output ONLY the refined prompt, no extra commentary."
    )

    client = get_client()
    try:
        chat = client.chat.completions.create(
            model=REFINER_MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": raw},
            ],
            temperature=0.2,
        )
        refined = (chat.choices[0].message.content or "").strip()
        return refined if refined else raw
    except Exception:
        log.exception("Refiner failed; falling back to raw.")
        return raw


def _chat_once(model: str, prompt: str) -> str:
    """
    Single Chat Completions call; returns plain text.
    """
    client = get_client()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        log.exception("LLM call failed")
        return f"[LLM error] {e}"


# ----------------------------
# Health & Static
# ----------------------------
@app.get("/health")
def health():
    return jsonify({"ok": True})


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


# ----------------------------
# Non-streaming endpoint
# ----------------------------
@app.post("/process-voice")
def process_voice():
    ip = (request.headers.get("x-forwarded-for") or request.remote_addr or "ip").split(",")[0].strip()
    if _too_many(ip):
        return jsonify({"error": "Too many requests"}), 429

    audio = request.files.get("audio")
    if not audio:
        return jsonify({"error": "No 'audio' file in form-data"}), 400

    model = (request.form.get("model") or "gpt-4o-mini").strip()
    mime = (request.form.get("mime") or audio.mimetype or "").lower()

    path = _save_upload_to_temp(audio, mime)
    try:
        # 1) STT
        raw_text = _transcribe_from_path(path)

        # 2) Refine (safe)
        refined_prompt = _refine(raw_text)

        # 3) LLM
        final = _chat_once(model, refined_prompt)

        return jsonify(
            {
                "raw": raw_text,
                "refined": refined_prompt,
                "response": final,
            }
        )
    except Exception as e:
        log.exception("process_voice failed")
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            os.remove(path)
        except Exception:
            pass


# ----------------------------
# Streaming endpoint (text/plain stream)
# ----------------------------
@app.post("/process-voice-stream")
def process_voice_stream():
    ip = (request.headers.get("x-forwarded-for") or request.remote_addr or "ip").split(",")[0].strip()
    if _too_many(ip):
        return jsonify({"error": "Too many requests"}), 429

    audio = request.files.get("audio")
    if not audio:
        return jsonify({"error": "No 'audio' file in form-data"}), 400

    model = (request.form.get("model") or "gpt-4o-mini").strip()
    mime = (request.form.get("mime") or audio.mimetype or "").lower()

    path = _save_upload_to_temp(audio, mime)

    def generate(refined_prompt: str):
        client = get_client()
        try:
            # Stream via Chat Completions
            stream = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": refined_prompt}],
                stream=True,
                temperature=0.7,
            )
            for chunk in stream:
                try:
                    delta = chunk.choices[0].delta
                    if delta and getattr(delta, "content", None):
                        yield delta.content
                except Exception:
                    # tolerate any partial chunks
                    continue
        except Exception as e:
            log.exception("Streaming LLM failed")
            yield f"\n[stream error] {e}"

    try:
        # 1) STT
        raw_text = _transcribe_from_path(path)

        # 2) Refine (safe)
        refined_prompt = _refine(raw_text)

        # 3) Build streaming response with custom headers
        resp = Response(stream_with_context(generate(refined_prompt)), mimetype="text/plain")
        resp.headers["X-Refined-Prompt"] = refined_prompt
        resp.headers["X-Transcript"] = raw_text
        resp.headers["Access-Control-Expose-Headers"] = "X-Refined-Prompt, X-Transcript"
        return resp

    except Exception as e:
        log.exception("process_voice_stream failed")
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            os.remove(path)
        except Exception:
            pass


# ----------------------------
# Local run (Railway uses gunicorn)
# ----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)
