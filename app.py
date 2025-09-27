from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os, tempfile
from openai import OpenAI

app = Flask(__name__, static_folder="static")
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25MB

_client = None
def get_client():
    global _client
    if _client is None:
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        _client = OpenAI(api_key=key)
    return _client

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/")
def root():
    return send_from_directory(app.static_folder, "index.html")

@app.post("/process-voice")
def process_voice():
    if "audio" not in request.files:
        return jsonify({"error": "No 'audio' file in form-data"}), 400
    up = request.files["audio"]

    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
        up.save(tmp.name)
        path = tmp.name

    try:
        client = get_client()

        # 1) Try direct English via translations
        try:
            transcript_text = client.audio.translations.create(
                model="whisper-1",
                file=open(path, "rb"),
                response_format="text",
                temperature=0
            ).strip()
        except Exception:
            # Fallback: STT then translate via Chat Completions
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

        # 2) Refine into a clean, actionable prompt
        refine = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content":
                 "You are a prompt engineer. Given the user's raw intent, output a single, clear, action-oriented prompt that an LLM can execute directly. Fix grammar, add obvious specifics, and keep it concise. Output ONLY the final prompt."},
                {"role": "user", "content": transcript_text}
            ],
            temperature=0
        )
        refined_prompt = refine.choices[0].message.content.strip()

        # 3) Execute the refined prompt
        final = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": refined_prompt}],
            temperature=0.2
        )
        final_text = final.choices[0].message.content.strip()

        return jsonify({"refined_prompt": refined_prompt, "response": final_text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        try: os.remove(path)
        except Exception: pass

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
