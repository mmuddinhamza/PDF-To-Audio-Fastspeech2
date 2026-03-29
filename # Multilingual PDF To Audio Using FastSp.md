# Multilingual PDF To Audio Using FastSpeech2_HS

This project is a web application that converts PDF files to audio using the FastSpeech2_HS model. It supports multiple languages.

## Features

*   Upload PDF files.
*   Convert PDF text to speech in multiple languages.
*   Generate and play audio files.

## Technologies Used

*   Python
*   Flask
*   FastSpeech2_HS
*   gunicorn
*   torch
*   espnet
*   pandas
*   matplotlib
*   indic-num2words

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd PDF-To-Audio-Fastspeech2
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    Alternatively, you can use conda with the `environment.yml` file:
    ```bash
    conda env create -f environment.yml
    conda activate <env_name>
    ```

## Usage

1.  **Run the Flask application:**
    ```bash
    python app_5.py
    ```
    *Note: Please verify the correct file to run the application. It is assumed to be `app_5.py`.*

2.  **Open your web browser and navigate to:**
    ```
    http://127.0.0.1:5000
    ```
    *Note: The port might be different. Please check the application's output or code for the correct URL.*

## Project Structure
```
.
├── api.py
├── app_5.py
├── environment.yml
├── get_phone_mapped_python.py
├── inference.py
├── multilingualcharmap.json
├── requirements.txt
├── text_preprocess_for_inference.py
├── generated_audio/
├── tmp/
├── uploaded_PDFs/
└── web_ui/
```
