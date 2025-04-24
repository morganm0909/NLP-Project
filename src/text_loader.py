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

import requests
from bs4 import BeautifulSoup

def load_text_from_url(url):
    """Fetch and clean text content from a webpage."""
    headers = {'User-Agent': 'Mozilla/5.0'}  # Pretend to be a browser
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        raise Exception(f"Failed to retrieve content from {url}, Status Code: {response.status_code}")

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove scripts, styles, and nav
    for tag in soup(["script", "style", "nav", "header", "footer", "noscript"]):
        tag.decompose()

    # Extract visible text
    text = soup.get_text(separator=' ', strip=True)

    # Collapse multiple spaces and newlines
    cleaned_text = ' '.join(text.split())

    if len(cleaned_text.split()) < 20:
        raise Exception("Retrieved content is too short or not suitable for summarization.")

    return cleaned_text
