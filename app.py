import os

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

from rag_pipeline import run_rag
from vector_store import store_document

# Create Flask application

app = Flask(
    __name__,
    static_folder="frontend",
    static_url_path=""
)

CORS(app)

# Serve frontend root

@app.route("/", methods=["GET"])
def index():
    return send_from_directory("frontend", "index.html")

# Upload folder

UPLOAD_FOLDER = os.path.join(
    app.root_path,
    "uploads"
)

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

# Health check

@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "status": "ok",
        "message": "DocLens AI backend is running"
    })


# Upload document-

@app.route("/upload", methods=["POST"])
def upload():

    # Check whether file exists

    if "file" not in request.files:

        return jsonify({
            "error": "PDF file is required"
        }), 400

    file = request.files["file"]

    # Check filename

    if file.filename == "":

        return jsonify({
            "error": "No file selected"
        }), 400

    # Check PDF extension

    if not file.filename.lower().endswith(".pdf"):

        return jsonify({
            "error": "Only PDF files are allowed"
        }), 400

    # Secure filename

    filename = secure_filename(
        file.filename
    )

    # Build absolute file path

    file_path = os.path.join(
        UPLOAD_FOLDER,
        filename
    )

    # Save uploaded PDF

    file.save(
        file_path
    )

    # Process document

    result = store_document(
        file_path
    )

    # Return document information

    return jsonify({
    "status": "success",
    "document_id": result["document_id"],
    "filename": filename,
    "pages": result["pages"],
    "chunks": result["chunks"],
    "duplicate": result.get("duplicate", False)
})


# Ask question

@app.route("/ask", methods=["POST"])
def ask():

    data = request.get_json()

    # Validate request body

    if not data:

        return jsonify({
            "error": "Request body is required"
        }), 400

    # Get document ID

    document_id = data.get(
        "document_id"
    )

    if not document_id:

        return jsonify({
            "error": "Document ID is required"
        }), 400

    # Get question

    question = data.get(
        "question"
    )

    if not question:

        return jsonify({
            "error": "Question is required"
        }), 400

    question = question.strip()

    if not question:

        return jsonify({
            "error": "Question cannot be empty"
        }), 400

    # Run RAG pipeline

    try:

        result = run_rag(
            question,
            document_id
        )

    except Exception as error:

        error_message = str(error)

        # Detect Groq rate-limit errors
        if "429" in error_message or "rate" in error_message.lower():

            return jsonify({
                "error": (
                    "The AI model rate limit has been reached. "
                    "Please wait a minute and try again."
                )
            }), 429

        return jsonify({
            "error": f"Pipeline error: {error_message}"
        }), 500

    # Return JSON response

    return jsonify(
        result
    )


# Start server

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )