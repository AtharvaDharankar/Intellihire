import os
import json
import fitz  # PyMuPDF
from pathlib import Path
import spacy
import random

# Load spacy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spacy model...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def extract_text_from_pdf(pdf_path):
    """Extracts all text from a PDF file."""
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text()
        return text.strip()
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return None

def extract_keywords(text, num_keywords=10):
    """Extracts nouns and proper nouns as keywords."""
    doc = nlp(text)
    keywords = [token.text for token in doc if token.pos_ in ['NOUN', 'PROPN'] and not token.is_stop and len(token.text) > 2]
    # Filter unique, case-insensitive
    unique_keywords = list(set([k.lower() for k in keywords]))
    # Pick random subset
    random.shuffle(unique_keywords)
    return unique_keywords[:num_keywords]

def generate_jd(keywords):
    """Generates a synthetic job description from keywords."""
    if not keywords:
        return "Looking for a professional with excellent communication skills and a strong work ethic."
    skills_str = ", ".join(keywords)
    return f"We are seeking an experienced candidate with strong skills in: {skills_str}. The ideal candidate will have hands-on experience and a proven track record in these areas."

def create_dynamic_dataset(pdf_folder, output_file):
    dataset = []
    pdf_files = list(Path(pdf_folder).glob("*.pdf"))
    
    print(f"Processing {len(pdf_files)} PDFs from {pdf_folder}...")
    
    resumes_data = []
    for pdf_path in pdf_files:
        text = extract_text_from_pdf(pdf_path)
        if text:
            keywords = extract_keywords(text, num_keywords=15)
            resumes_data.append({
                "name": pdf_path.name,
                "text": text,
                "keywords": keywords
            })
            print(f"  [OK] Processed: {pdf_path.name}")

    if len(resumes_data) < 2:
        print("Need at least 2 resumes to generate negative matches.")
        return

    # Generate pairs
    for i, res in enumerate(resumes_data):
        # 1. Positive Match
        positive_jd = generate_jd(res["keywords"][:8])
        dataset.append({
            "texts": [res["text"], positive_jd],
            "label": 1.0
        })

        # 2. Negative Match
        # Pick a different resume's keywords
        other_idx = (i + 1) % len(resumes_data)
        negative_jd = generate_jd(resumes_data[other_idx]["keywords"][:8])
        dataset.append({
            "texts": [res["text"], negative_jd],
            "label": 0.0
        })

        # 3. Partial Match
        # Mix 4 keywords from candidate and 4 from someone else
        mixed_keywords = res["keywords"][8:12] + resumes_data[other_idx]["keywords"][8:12]
        partial_jd = generate_jd(mixed_keywords)
        dataset.append({
            "texts": [res["text"], partial_jd],
            "label": 0.5
        })

    # Save to JSON
    # we'll completely overwrite the training data so we don't have stale bad data
    with open(output_file, "w") as f:
        json.dump(dataset, f, indent=4)
    
    print(f"\nSuccess! Generated {len(dataset)} pairs and saved to {output_file}")

if __name__ == "__main__":
    # --- CONFIGURATION ---
    # Put your PDF resumes in this folder
    RESUME_FOLDER = r"c:\Resume_model\resumes_to_train"
    OUTPUT_JSON = r"c:\Resume_model\training_data.json"
    # ---------------------

    if not os.path.exists(RESUME_FOLDER):
        print(f"Error: Folder '{RESUME_FOLDER}' not found.")
    else:
        create_dynamic_dataset(RESUME_FOLDER, OUTPUT_JSON)