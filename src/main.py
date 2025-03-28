from text_loader import load_text_from_file, load_text_from_pdf, load_text_from_url
from preprocess import clean_text
from summarizer import summarize_text
from keypoints import extract_key_points
import nltk

def main():
    # Choose input method
    source = input("Enter source type (file/pdf/url): ").strip().lower()
    
    if source == "file":
        filepath = input("Enter file path: ").strip()
        input_text = load_text_from_file(filepath)
    elif source == "pdf":
        pdf_path = input("Enter PDF file path: ").strip()
        input_text = load_text_from_pdf(pdf_path)
    elif source == "url":
        url = input("Enter URL: ").strip()
        input_text = load_text_from_url(url)
    else:
        print("Invalid source type. Please enter 'file', 'pdf', or 'url'.")
        return

    # Preprocess text
    cleaned_text = clean_text(input_text)

    # Summarize text
    summary = summarize_text(cleaned_text)

    # Extract key points
    key_points = extract_key_points(cleaned_text)

    print("\nSummary:\n", summary)
    print("\nKey Points:\n", key_points)

if __name__ == "__main__":
    main()
