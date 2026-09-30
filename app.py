import os
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
                    f"Loading Whisper model: "
                    f"{WHISPER_MODEL_NAME}"
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
# AUDIO SUMMARY JOB FUNCTION
# ---------------------------------------------------------

AUDIO_SUMMARY_JOB_URL = os.environ.get(
    "AUDIO_SUMMARY_JOB_URL",
    (
        "https://audiosummaryagent-60086819444."
        "development.catalystserverless.in/"
        "server/AudioSummaryZiaTrigger/"
    )
)


# ---------------------------------------------------------
# AUDIO SUMMARY STATUS FUNCTION
# ---------------------------------------------------------

AUDIO_SUMMARY_STATUS_URL = os.environ.get(
    "AUDIO_SUMMARY_STATUS_URL",
    (
        "https://audiosummaryagent-60086819444."
        "development.catalystserverless.in/"
        "server/AudioSummaryJobStatus/"
    )
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
        "summary_pipeline":
            "enabled"
    })


# ---------------------------------------------------------
# SUMMARY STATUS
# ---------------------------------------------------------

@app.route("/summary-status/<job_id>", methods=["GET"])
def summary_status(job_id):

    try:

        response = requests.get(
            AUDIO_SUMMARY_STATUS_URL,
            params={
                "job_id": job_id
            },
            timeout=20
        )

        try:
            data = response.json()

        except ValueError:
            return jsonify({
                "status": "error",
                "message":
                    "Invalid response from "
                    "AudioSummaryJobStatus."
            }), 502

        return jsonify(data), response.status_code

    except requests.Timeout:

        return jsonify({
            "status": "error",
            "message":
                "Summary status request timed out."
        }), 504

    except requests.RequestException as error:

        return jsonify({
            "status": "error",
            "message":
                "Could not retrieve summary status.",
            "details": str(error)
        }), 502


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
        # TRANSCRIBE AUDIO
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
        # NO SPEECH FOUND
        # -------------------------------------------------

        if not transcript:

            return jsonify({
                "status":
                    "success",

                "language":
                    language,

                "transcript":
                    "",

                "message":
                    "No speech was detected "
                    "in the recording."
            })


        print(
            "Transcription completed successfully"
        )

        print(
            f"Transcript length: "
            f"{len(transcript)}"
        )


        # -------------------------------------------------
        # CREATE AUDIO SUMMARY JOB
        # -------------------------------------------------

        job_payload = {
            "language":
                language,

            "transcript":
                transcript
        }


        job_id = None
        job_status = None
        summary_job_message = None


        try:

            print(
                "Creating AudioSummaryJobs job"
            )


            job_response = requests.post(
                AUDIO_SUMMARY_JOB_URL,
                json=job_payload,
                timeout=20
            )


            print(
                "AudioSummaryZiaTrigger "
                "HTTP status:",
                job_response.status_code
            )


            if job_response.status_code in [
                200,
                201,
                202
            ]:

                try:

                    job_data = (
                        job_response.json()
                    )

                    job_id = (
                        job_data.get(
                            "job_id"
                        )
                    )

                    job_status = (
                        job_data.get(
                            "job_status",
                            "pending"
                        )
                    )

                    summary_job_message = (
                        job_data.get(
                            "message"
                        )
                    )


                    print(
                        "Summary job created:",
                        job_id
                    )


                except ValueError:

                    summary_job_message = (
                        "Summary job was created, "
                        "but the response could "
                        "not be parsed."
                    )


            else:

                summary_job_message = (
                    "Transcription succeeded, "
                    "but summary job creation failed."
                )

                print(
                    "Summary job creation failed:",
                    job_response.text
                )


        except requests.Timeout:

            summary_job_message = (
                "Transcription succeeded, "
                "but summary job creation timed out."
            )

            print(
                "AudioSummaryZiaTrigger "
                "request timed out"
            )


        except requests.RequestException as error:

            summary_job_message = (
                "Transcription succeeded, "
                "but summary job could "
                "not be created."
            )

            print(
                "Summary job request error:",
                str(error)
            )


        # -------------------------------------------------
        # RETURN TRANSCRIPT + JOB DETAILS
        # -------------------------------------------------

        return jsonify({
            "status":
                "success",

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
                ),

            "summary_job": {
                "job_id":
                    job_id,

                "status":
                    job_status,

                "message":
                    summary_job_message
            }
        })


    # -----------------------------------------------------
    # TRANSCRIPTION ERROR
    # -----------------------------------------------------

    except Exception as error:

        print(
            "Transcription error:",
            str(error)
        )


        return jsonify({
            "status":
                "error",

            "message":
                str(error)
        }), 500


    # -----------------------------------------------------
    # CLEAN TEMP FILE
    # -----------------------------------------------------

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
