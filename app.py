import os
import tempfile
import threading

from flask import Flask, jsonify, request, send_from_directory
from faster_whisper import WhisperModel


app = Flask(__name__)


# ---------------------------------------------------------
# WHISPER CONFIGURATION
# ---------------------------------------------------------

WHISPER_MODEL_NAME = os.environ.get(
    "WHISPER_MODEL",
    "large-v3-turbo"
)

whisper_model = None
whisper_lock = threading.Lock()


def get_whisper_model():
    """
    Load Whisper only when the first transcription request arrives.

    We do this lazily so AppSail can start the Flask server first
    instead of waiting for the large model during startup.
    """

    global whisper_model

    if whisper_model is None:
        with whisper_lock:
            if whisper_model is None:
                print(
                    f"Loading Whisper model: {WHISPER_MODEL_NAME}"
                )

                whisper_model = WhisperModel(
                    WHISPER_MODEL_NAME,
                    device="cpu",
                    compute_type="int8"
                )

                print("Whisper model loaded successfully")

    return whisper_model

@app.route("/", methods=["GET"])
def home():
    return send_from_directory(
        ".",
        "index.html"
    )

# ---------------------------------------------------------
# HEALTH CHECK
# ---------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "success",
        "message": "Audio Summary backend is running",
        "model": WHISPER_MODEL_NAME,
        "supported_languages": [
            "ja",
            "zh"
        ]
    })


# ---------------------------------------------------------
# TRANSCRIPTION
# ---------------------------------------------------------

@app.route("/transcribe", methods=["POST"])
def transcribe():
    temp_path = None

    try:
        # Check audio file
        if "audio" not in request.files:
            return jsonify({
                "status": "error",
                "message": "Please upload an MP3 file."
            }), 400

        audio_file = request.files["audio"]

        if not audio_file.filename:
            return jsonify({
                "status": "error",
                "message": "No audio file was selected."
            }), 400

        # Language must be Japanese or Chinese
        language = (
            request.form.get("language", "")
            .strip()
            .lower()
        )

        if language not in ["ja", "zh"]:
            return jsonify({
                "status": "error",
                "message":
                    "Language must be 'ja' for Japanese "
                    "or 'zh' for Chinese."
            }), 400

        # Save uploaded MP3 temporarily
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp3"
        ) as temp_audio:

            audio_file.save(temp_audio.name)

            temp_path = temp_audio.name

        print(
            f"Transcribing file: "
            f"{audio_file.filename}, "
            f"language: {language}"
        )

        # Load model
        model = get_whisper_model()

        # Transcribe
        segments, info = model.transcribe(
    temp_path,
    language=language,
    task="transcribe",
    beam_size=5,
    temperature=0,
    vad_filter=True
)

        transcript_parts = []

        for segment in segments:
            text = segment.text.strip()

            if text:
                transcript_parts.append(text)

        transcript = " ".join(
            transcript_parts
        ).strip()

        if not transcript:
            return jsonify({
                "status": "success",
                "language": language,
                "transcript": "",
                "message":
                    "No speech was detected in the recording."
            })

        return jsonify({
            "status": "success",
            "file_name": audio_file.filename,
            "language": language,
            "transcript": transcript,
            "duration_seconds":
                getattr(
                    info,
                    "duration",
                    None
                )
        })

    except Exception as e:
        print(
            "Transcription error:",
            str(e)
        )

        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

    finally:
        if (
            temp_path
            and os.path.exists(temp_path)
        ):
            try:
                os.remove(temp_path)
            except Exception:
                pass


# ---------------------------------------------------------
# START SERVER
# ---------------------------------------------------------

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "X_ZOHO_CATALYST_LISTEN_PORT",
            "9000"
        )
    )

    print(
        f"Starting Audio Summary backend "
        f"on port {port}"
    )

    app.run(
        host="0.0.0.0",
        port=port
    )
