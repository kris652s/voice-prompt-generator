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

# ----------------------------
# App / Config
# ----------------------------
app = Flask(__name__, static_folder="static")
ALLOWED_ORIGIN = os.getenv("ALLOWED_ORIGIN", "*")
CORS(app, resources={r"/*": {"origins": ALLOWED_ORIGIN}})
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25 MB

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("voice-prompt")

client = OpenAI()  # uses OPENAI_API_KEY from env
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
def _require_audio():
    audio_file = request.files.get("audio")
    if not audio_file:
        return None, (jsonify({"error": "No 'audio' file in form-data"}), 400)
    return audio_file, None


def _transcribe(audio_file) -> str:
    """
    Whisper (server-side). Returns the transcribed text.
    """
    tr = client.audio.transcriptions.create(model="whisper-1", file=audio_file)
    text = (tr.text or "").strip()
    return text


def _refine(raw_text: str) -> str:
    """
    Safe prompt refiner:
    - Keep user's original meaning exactly.
    - If not English, translate to clear natural English.
    - Remove filler words, keep key details.
    - Output ONLY the refined prompt (no preamble, no quotes).
    """
    raw = (raw_text or "").strip()
    if not raw:
        return raw

    instruction = (
        "Rewrite the user's utterance into a clear, concise prompt for an LLM.\n"
        "Requirements:\n"
        "- Preserve the user's original meaning exactly.\n"
        "- If the text is not in English, translate it to natural English.\n"
        "- Remove filler words and hesitations, keep essential details.\n"
        "- Do NOT invent details or add assumptions.\n"
        "- Output ONLY the refined prompt, no extra commentary."
    )

    try:
        resp = client.responses.create(
            model=REFINER_MODEL,
            input=f"{instruction}\n\nUser utterance:\n{raw}",
        )
        refined = (resp.output_text or "").strip()
        # Fallback if the model returned nothing
        return refined if refined else raw
    except Exception as e:
        log.exception("Refiner failed")
        return raw


def _chat_once(model: str, prompt: str) -> str:
    """
    Single Responses API call; returns plain text.
    """
    try:
        resp = client.responses.create(model=model, input=prompt)
        return (resp.output_text or "").strip()
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
    ip = request.headers.get("x-forwarded-for", request.remote_addr or "ip")
    if _too_many(ip):
        return jsonify({"error": "Too many requests"}), 429

    audio_file, err = _require_audio()
    if err:
        return err

    model = request.form.get("model", "gpt-4o-mini")

    # 1) STT
    raw_text = _transcribe(audio_file)

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


# ----------------------------
# Streaming endpoint (text/plain stream)
# ----------------------------
@app.post("/process-voice-stream")
def process_voice_stream():
    ip = request.headers.get("x-forwarded-for", request.remote_addr or "ip")
    if _too_many(ip):
        return jsonify({"error": "Too many requests"}), 429

    audio_file, err = _require_audio()
    if err:
        return err

    model = request.form.get("model", "gpt-4o-mini")

    # 1) STT
    raw_text = _transcribe(audio_file)

    # 2) Refine (safe)
    refined_prompt = _refine(raw_text)

    # 3) Stream LLM output
    def generate():
        try:
            with client.responses.stream(model=model, input=refined_prompt) as stream:
                for event in stream:
                    if getattr(event, "type", "") == "response.output_text.delta":
                        chunk = getattr(event, "delta", "")
                        if chunk:
                            yield chunk
                stream.close()
        except Exception as e:
            log.exception("Streaming LLM failed")
            yield f"\n[stream error] {e}"

    resp = Response(stream_with_context(generate()), mimetype="text/plain")
    # Let the frontend read the refined text while streaming
    resp.headers["X-Refined-Prompt"] = refined_prompt
    resp.headers["X-Transcript"] = raw_text
    return resp


# ----------------------------
# Local run (Railway uses gunicorn)
# ----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)
