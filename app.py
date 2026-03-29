from flask import Flask, request, send_file, jsonify, send_from_directory
import subprocess
import os
import base64
import PyPDF2
from io import BytesIO
import logging
from flask_cors import CORS


app = Flask(__name__)
CORS(app) # Enable CORS for all routes


#Configure logging
logging.basicConfig(level=logging.ERROR)


# Set the absolute path to the models folder
MODEL_ROOT = os.path.expanduser("~/PDF-To-Audio-Fastspeech2/models")


#Directory configurations
UPLOAD_PDF_DIR = "uploaded_PDFs"
GENERATED_AUDIO_DIR = "generated_audio"
os.makedirs(UPLOAD_PDF_DIR, exist_ok=True)
os.makedirs(GENERATED_AUDIO_DIR, exist_ok=True)



def extract_text_from_pdf(base64_string):

    try:
        pdf_bytes = base64.b64decode(base64_string)
        pdf_file = BytesIO(pdf_bytes)
        reader = PyPDF2.PdfReader(pdf_file)

        text = "\n".join([page.extract_text() or "" for page in reader.pages])

        if not text.strip():
            raise ValueError("No extractable text found in PDF.")
        

        # Remove null bytes
        text = text.replace("\x00", "")

        return text

    except Exception as e:

        logging.error(f"Error processing PDF: {str(e)}", exc_info=True)
        raise ValueError(f"Error processing PDF: {str(e)}")


def save_pdf_file(base64_string, filename):
    try:

        pdf_bytes = base64.b64decode(base64_string)
        file_path = os.path.join(UPLOAD_PDF_DIR, filename)

        with open(file_path, 'wb') as f:
            f.write(pdf_bytes)

        return file_path

    except Exception as e:

        logging.error(f"Error saving PDF: {str(e)}")
        raise ValueError(f"Error saving PDF: {str(e)}")
    


@app.route("/generate_audio", methods=["POST"])
def generate_audio():
    try:

        data = request.get_json()

        if not data:
            return jsonify({"error": "Invalid, request. Missing data."}), 400
        

        text = ""
        pdf_base64 = data.get("pdf_base64")
        language = data.get("language", "english")
        gender = data.get("gender", "male")
        alpha = data.get("alpha", "1")
        filename = data.get("filename")


        if not pdf_base64:
            return jsonify({"error": "No PDF provided"}), 400

        if not filename:
            return jsonify({"error": "No filename provided."}), 400

        #Ensure filename ends with .pdf
        if not filename.endswith('.pdf'):
            filename += '.pdf'

          # Check if file already exists
        pdf_path = os.path.join(UPLOAD_PDF_DIR, filename)
        if os.path.exists(pdf_path):
            return jsonify({"error": f"File '{filename}' already exists on server."}), 400


        try:
            # save the PDF file
            pdf_path = save_pdf_file(pdf_base64, filename)

            # Extract text from PDF
            text = extract_text_from_pdf(pdf_base64)

        except ValueError as e:
            #Clean up if we saved the PDF but failed later
            if os.path.exists(pdf_path):
                os.remove(pdf_path)

            return jsonify({"error": str(e)}), 400


        if not text:

            # Clean up if we saved the PDF but got no text
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
            return jsonify({"error": "No valid text provided."}), 400


        # Generate matching audio filename
        audio_filename = filename.replace(".pdf", ".wav")
        output_path = os.path.join(GENERATED_AUDIO_DIR, audio_filename)


        cmd = [
            "python", "inference.py",
            "--sample_text", text,
            "--language", language,
            "--gender", gender,
            "--alpha", alpha,
            "--output_file", output_path,
            "--model_root", MODEL_ROOT
        ]

        try:
            subprocess.run(cmd, check=True)
            
        except subprocess.CalledProcessError as e:

            #clean both files if audio generation fails
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
            if os.path.exists(output_path):
                os.remove(output_path)

            logging.error(f"Audio generation failed: {str(e)}")

            return jsonify({"error": "Audio generation failed."}), 500
        
        return jsonify({
            "message": "Audio generated successfully",
            "pdf_filename": filename,
            "audio_filename": audio_filename
        })
    except Exception as e:

        logging.error(f"General error: {str(e)}")
        return jsonify({"error": str(e)}), 500
    


@app.route("/get_pdf/<filename>", methods=["GET"])
def get_pdf(filename):
    try:
        if not filename.endswith('.pdf'):
            filename += '.pdf'

        file_path = os.path.join(UPLOAD_PDF_DIR, filename)

        if not os.path.exists(file_path):
            return jsonify({"error": "PDF not found."}), 404
        
        return send_file(file_path, as_attachment=True, mimetype='application/pdf')
    
    except Exception as e:
        logging.error(f"Error retrieving PDF: {str(e)}")

        return jsonify({"error": str(e)}), 500



@app.route("/get_audio/<pdf_filename>", methods=["GET"])
def get_audio(pdf_filename):
    try:
        if not pdf_filename.endswith('.pdf'):
            pdf_filename += '.pdf'

        # Convert PDF filename to audio filename
        audio_filename = pdf_filename.replace(".pdf", ".wav")
        file_path = os.path.join(GENERATED_AUDIO_DIR, audio_filename)

        if not os.path.exists(file_path):
            return jsonify({"error": "Audio file not found."}), 404
        
        return send_file(file_path, as_attachment=True, mimetype='audio/wav')
    
    except Exception as e:

        logging.error(f"Error retrieving audio: {str(e)}")
        return jsonify({"error": str(e)}), 500



@app.route("/list_files", methods=["GET"])
def list_files():
    try:

        pdf_files = [f for f in os.listdir(UPLOAD_PDF_DIR) if f.endswith('.pdf')]
        audio_files = [f for f in os.listdir(GENERATED_AUDIO_DIR) if f.endswith('.wav')]

        return jsonify({
            "pdf_files": pdf_files,
            "audio_files": audio_files
        })
    except Exception as e:

        logging.error(f"Error listing files: {str(e)}")
        return jsonify({"error": str(e)}), 500



# Serve frontend at root URL
@app.route("/")
def serve_frontend():

    return send_from_directory("web_ui", "web_ui_2.html")



if __name__ == "__main__":

    app.run(host='0.0.0.0', port=5000, debug=True)