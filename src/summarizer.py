from transformers import pipeline

def summarize_text(text):
    """Use a pre-trained model to summarize text."""
    max_length = len(text) // 2
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(text, max_length=max_length, min_length=0, do_sample=False)
    return summary[0]['summary_text']
