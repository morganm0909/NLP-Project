from transformers import pipeline

def summarize_text(text, max_length=150):
    """Use a pre-trained model to summarize text."""
    summarizer = pipeline("summarization")
    summary = summarizer(text, max_length=max_length, min_length=50, do_sample=False)
    return summary[0]['summary_text']
