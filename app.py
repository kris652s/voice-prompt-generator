 from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os, tempfile
from openai import OpenAI

app = Flask(__name__, static_folder="static")
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25MB

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set")
client = OpenAI(api_key=OPENAI_API_KEY)

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
        # ----- 1) STT with English output -----
        try:
            # Preferred: direct English via translations
            transcript_text = client.audio.translations.create(
                model="whisper-1",
                file=open(path, "rb"),
                response_format="text",
                temperature=0
            ).strip()
        except Exception:
            # Fallback: plain transcription + LLM translation to English
            raw = client.audio.transcriptions.create(
                model="whisper-1",
                file=open(path, "rb"),
                response_format="text",
                temperature=0
            ).strip()
            trans = client.responses.create(
                model="gpt-4o-mini",
                input=f"Translate to natural English, preserve meaning:\n\n{raw}"
            )
            transcript_text = trans.output_text.strip()

        # ----- 2) Refine into a clean, actionable prompt -----
        refine_instructions = (
            "You are a prompt engineer. Given the raw user intent below, "
            "output a single, clear, action-oriented LLM prompt. "
            "Fix grammar, add obvious specifics, and keep it concise. "
            "Output ONLY the final prompt.\n\n"
            f"Raw user intent:\n{transcript_text}"
        )
        refined = client.responses.create(
            model="gpt-4o-mini",
            input=refine_instructions
        )
        refined_prompt = refined.output_text.strip()

        # ----- 3) Execute the refined prompt -----
        final = client.responses.create(
            model="gpt-4o-mini",
            input=refined_prompt
        )
        final_text = final.output_text.strip()

        return jsonify({"refined_prompt": refined_prompt, "response": final_text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            os.remove(path)
        except Exception:
            pass

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
