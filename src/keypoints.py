import spacy

nlp = spacy.load("en_core_web_sm")

def extract_key_points(text, num_points=5):
    """Extract key points from text using Named Entity Recognition (NER)."""
    doc = nlp(text)
    key_points = set()

    for ent in doc.ents:
        key_points.add(ent.text)

    return list(key_points)[:num_points]
