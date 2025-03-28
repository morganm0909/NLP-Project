import os
import requests
import PyPDF2

def load_text_from_file(filepath):
    """Load text from a .txt file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File {filepath} not found.")
    
    with open(filepath, "r", encoding="utf-8") as file:
        return file.read()

def load_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"File {pdf_path} not found.")
    
    text = ""
    with open(pdf_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    
    return text.strip()

def load_text_from_url(url):
    """Fetch text content from a webpage."""
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        raise Exception(f"Failed to retrieve content from {url}, Status Code: {response.status_code}")
