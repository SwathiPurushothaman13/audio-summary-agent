import os
import json
import tempfile
import threading

import requests
from flask import Flask, jsonify, request, send_from_directory
from faster_whisper import WhisperModel


app = Flask(__name__)


# ---------------------------------------------------------
# WHISPER CONFIGURATION
# ---------------------------------------------------------

WHISPER_MODEL_NAME = os.environ.get(
    "WHISPER_MODEL",
    "small"
)

whisper_model = None
whisper_lock = threading.Lock()


def get_whisper_model():
    """
    Load Whisper only when the first transcription request arrives.
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

                print(
                    "Whisper model loaded successfully"
                )

    return whisper_model


# ---------------------------------------------------------
# ZIA AGENT CONFIGURATION
# ---------------------------------------------------------

ZIA_AGENT_ID = "2175000000268001"

ZIA_ORG_ID = "60076591677"

ZIA_ACCESS_TOKEN = os.environ.get(
    "ZIA_ACCESS_TOKEN"
)

ZIA_TRIGGER_URL = (
    "https://agents.zoho.in/"
    "ziaagents/api/v1/agents/"
    + ZIA_AGENT_ID
    + "/trigger"
)


# ---------------------------------------------------------
# HOME
# ---------------------------------------------------------

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
        "message":
            "Audio Summary backend is running",
        "model":
            WHISPER_MODEL_NAME,
        "supported_languages": [
            "ja",
            "zh"
        ],
        "zia_agent_configured":
            bool(ZIA_ACCESS_TOKEN)
    })


# ---------------------------------------------------------
# TRANSCRIPTION
# ---------------------------------------------------------

@app.route("/transcribe", methods=["POST"])
def transcribe():
    temp_path = None

    try:

        # -------------------------------------------------
        # CHECK AUDIO FILE
        # -------------------------------------------------

        if "audio" not in request.files:
            return jsonify({
                "status": "error",
                "message":
                    "Please upload an MP3 file."
            }), 400

        audio_file = request.files["audio"]

        if not audio_file.filename:
            return jsonify({
                "status": "error",
                "message":
                    "No audio file was selected."
            }), 400


        # -------------------------------------------------
        # CHECK LANGUAGE
        # -------------------------------------------------

        language = (
            request.form.get(
                "language",
                ""
            )
            .strip()
            .lower()
        )

        if language not in [
            "ja",
            "zh"
        ]:
            return jsonify({
                "status": "error",
                "message":
                    "Language must be 'ja' "
                    "for Japanese or 'zh' "
                    "for Chinese."
            }), 400


        # -------------------------------------------------
        # SAVE MP3 TEMPORARILY
        # -------------------------------------------------

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp3"
        ) as temp_audio:

            audio_file.save(
                temp_audio.name
            )

            temp_path = (
                temp_audio.name
            )


        print(
            f"Transcribing file: "
            f"{audio_file.filename}, "
            f"language: {language}"
        )


        # -------------------------------------------------
        # LOAD WHISPER
        # -------------------------------------------------

        model = get_whisper_model()


        # -------------------------------------------------
        # TRANSCRIBE
        # -------------------------------------------------

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

            text = (
                segment.text
                .strip()
            )

            if text:
                transcript_parts.append(
                    text
                )


        transcript = " ".join(
            transcript_parts
        ).strip()


        # -------------------------------------------------
        # NO SPEECH
        # -------------------------------------------------

        if not transcript:
            return jsonify({
                "status": "success",
                "language":
                    language,
                "transcript":
                    "",
                "message":
                    "No speech was detected "
                    "in the recording."
            })


        # -------------------------------------------------
        # RETURN TRANSCRIPT
        # -------------------------------------------------

        return jsonify({
            "status": "success",
            "file_name":
                audio_file.filename,
            "language":
                language,
            "transcript":
                transcript,
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
            "message":
                str(e)
        }), 500


    finally:

        if (
            temp_path
            and os.path.exists(
                temp_path
            )
        ):
            try:
                os.remove(
                    temp_path
                )

            except Exception:
                pass


# ---------------------------------------------------------
# ZIA AGENT SUMMARY
# ---------------------------------------------------------

@app.route("/summarize", methods=["POST"])
def summarize():

    try:

        # -------------------------------------------------
        # READ INPUT
        # -------------------------------------------------

        data = (
            request.get_json(
                silent=True
            )
            or {}
        )


        language = (
            data.get(
                "language",
                ""
            )
            .strip()
            .lower()
        )


        transcript = (
            data.get(
                "transcript",
                ""
            )
            .strip()
        )


        # -------------------------------------------------
        # VALIDATE LANGUAGE
        # -------------------------------------------------

        if language not in [
            "ja",
            "zh"
        ]:
            return jsonify({
                "status": "error",
                "message":
                    "Language must be "
                    "'ja' or 'zh'."
            }), 400


        # -------------------------------------------------
        # VALIDATE TRANSCRIPT
        # -------------------------------------------------

        if not transcript:
            return jsonify({
                "status": "error",
                "message":
                    "Transcript is required."
            }), 400


        # -------------------------------------------------
        # CHECK TOKEN
        # -------------------------------------------------

        if not ZIA_ACCESS_TOKEN:
            return jsonify({
                "status": "error",
                "message":
                    "ZIA_ACCESS_TOKEN "
                    "is not configured."
            }), 500


        # -------------------------------------------------
        # PREPARE AGENT INPUT
        # -------------------------------------------------

        agent_input = {
            "language":
                language,
            "transcript":
                transcript
        }


        query_text = json.dumps(
            agent_input,
            ensure_ascii=False
        )


        payload = {
            "query":
                query_text,
            "systemArgs":
                {},
            "reasoning":
                False,
            "attachments":
                []
        }


        # -------------------------------------------------
        # HEADERS
        # -------------------------------------------------

        headers = {
            "Authorization":
                f"Zoho-oauthtoken "
                f"{ZIA_ACCESS_TOKEN}",

            "X-ZIAAGENTS-ORG":
                ZIA_ORG_ID,

            "Content-Type":
                "application/json"
        }


        print(
            "Sending transcript to "
            "Zia Agent"
        )

        print(
            f"Language: {language}"
        )

        print(
            f"Transcript length: "
            f"{len(transcript)}"
        )


        # -------------------------------------------------
        # CALL ZIA AGENT
        # -------------------------------------------------

        zia_response = requests.post(
            ZIA_TRIGGER_URL,
            headers=headers,
            json=payload,
            timeout=120
        )


        print(
            "Zia HTTP status:",
            zia_response.status_code
        )


        # -------------------------------------------------
        # PARSE API RESPONSE
        # -------------------------------------------------

        try:

            zia_data = (
                zia_response.json()
            )

        except ValueError:

            print(
                "Invalid Zia response:",
                zia_response.text
            )

            return jsonify({
                "status": "error",
                "message":
                    "Zia Agent returned "
                    "an invalid response."
            }), 502


        # -------------------------------------------------
        # HTTP FAILURE
        # -------------------------------------------------

        if (
            zia_response.status_code
            != 200
        ):

            print(
                "Zia Agent API error:",
                zia_data
            )

            return jsonify({
                "status": "error",
                "message":
                    "Zia Agent request "
                    "failed.",
                "zia_response":
                    zia_data
            }), zia_response.status_code


        # -------------------------------------------------
        # AGENT FAILURE
        # -------------------------------------------------

        if (
            zia_data.get("status")
            != "success"
        ):

            print(
                "Zia execution error:",
                zia_data
            )

            return jsonify({
                "status": "error",
                "message":
                    "Zia Agent execution "
                    "failed.",
                "zia_response":
                    zia_data
            }), 502


        # -------------------------------------------------
        # GET EXECUTION DATA
        # -------------------------------------------------

        execution_data = (
            zia_data.get(
                "data",
                {}
            )
        )


        raw_agent_response = (
            execution_data
            .get(
                "response",
                ""
            )
            .strip()
        )


        if not raw_agent_response:

            return jsonify({
                "status": "error",
                "message":
                    "Zia Agent returned "
                    "an empty response."
            }), 502


        print(
            "Zia Agent response received"
        )


        # -------------------------------------------------
        # PARSE AGENT JSON
        # -------------------------------------------------

        try:

            summary_data = json.loads(
                raw_agent_response
            )

        except json.JSONDecodeError:

            print(
                "Could not parse "
                "agent response:",
                raw_agent_response
            )

            return jsonify({
                "status": "error",
                "message":
                    "Unable to parse "
                    "Zia Agent summary.",
                "raw_response":
                    raw_agent_response
            }), 502


        # -------------------------------------------------
        # RETURN SUMMARY
        # -------------------------------------------------

        return jsonify({
            "status":
                "success",

            "language":
                language,

            "transcript":
                transcript,

            "summary":
                summary_data.get(
                    "summary",
                    ""
                ),

            "key_points":
                summary_data.get(
                    "key_points",
                    []
                ),

            "action_items":
                summary_data.get(
                    "action_items",
                    []
                ),

            "execution_id":
                execution_data.get(
                    "executionId"
                ),

            "session_id":
                execution_data.get(
                    "sessionId"
                )
        })


    # -----------------------------------------------------
    # ZIA TIMEOUT
    # -----------------------------------------------------

    except requests.Timeout:

        print(
            "Zia Agent request timed out"
        )

        return jsonify({
            "status": "error",
            "message":
                "Zia Agent request "
                "timed out."
        }), 504


    # -----------------------------------------------------
    # OTHER ZIA ERRORS
    # -----------------------------------------------------

    except requests.RequestException as e:

        print(
            "Zia request error:",
            str(e)
        )

        return jsonify({
            "status": "error",
            "message":
                "Unable to connect "
                "to Zia Agent.",
            "error":
                str(e)
        }), 502


    except Exception as e:

        print(
            "Summary error:",
            str(e)
        )

        return jsonify({
            "status": "error",
            "message":
                str(e)
        }), 500


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
        f"Starting Audio Summary "
        f"backend on port {port}"
    )

    app.run(
        host="0.0.0.0",
        port=port
    )
