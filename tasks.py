"""
Background task definitions for RQ (Redis Queue).
Handles resume parsing and ranking asynchronously so Flask never times out.
"""

import os
import re
import json
import spacy
import PyPDF2
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load spaCy NER model once per worker process
nlp = spacy.load("en_core_web_sm")


# --- Text Extraction (same logic as app.py, duplicated for worker isolation) ---

def extract_text_from_pdf(file_path):
    """Extract text from a PDF file."""
    with open(file_path, "rb") as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        return text


def extract_text_from_docx(file_path):
    """Extract text from a DOCX file."""
    import docx
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])


def extract_text_from_txt(file_path):
    """Extract text from a plain text file."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def extract_text(file_path):
    """Extract text based on file extension."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext == ".docx":
        return extract_text_from_docx(file_path)
    elif ext == ".txt":
        return extract_text_from_txt(file_path)
    else:
        return ""


def extract_entities(text):
    """Extract entities using regex patterns."""
    emails = re.findall(r'\S+@\S+', text)
    names = re.findall(r'^([A-Z][a-z]+)\s+([A-Z][a-z]+)', text)
    if names:
        names = [" ".join(names[0])]
    return emails, names


def extract_experience(text):
    """Extract total years of experience from resume text.

    Looks for year-range patterns such as:
      2018-2023, 2015 - present, 2019 – Current, 2020-till date, 2017-now
    Sums all non-overlapping durations and returns integer years.
    Returns 0 when no ranges are found (treated as 'Fresher').
    """
    current_year = datetime.now().year
    pattern = (
        r'((?:19|20)\d{2})'
        r'\s*[-–—]\s*'
        r'((?:19|20)\d{2}|[Pp]resent|[Cc]urrent|[Tt]ill\s*[Dd]ate|[Nn]ow)'
    )
    matches = re.findall(pattern, text)
    if not matches:
        return 0

    total = 0
    for start_str, end_str in matches:
        start_year = int(start_str)
        end_year = current_year if not end_str.strip().isdigit() else int(end_str)
        diff = end_year - start_year
        if 0 < diff <= 50:
            total += diff
    return total


# --- Background Task ---

def analyze_resumes_task(job_description, file_paths, results_path, ranking_preference='skillset', skill_weights=None):
    """
    Background worker function that performs NLP on resumes and saves results.
    The actual heavy work: parse each resume, compute TF-IDF similarity, rank.
    Writes progress updates and the final result to a JSON file so the Flask
    server can poll it.

    Args:
        job_description: The job description text.
        file_paths: List of absolute paths to uploaded resume files.
        results_path: Path to the JSON file where results/progress are written.
        ranking_preference: Sort strategy.
        skill_weights: Dict of skill:weight pairs (1-10) for scoring bonus.
    """
    if skill_weights is None:
        skill_weights = {}

    total = len(file_paths)

    # Write initial progress
    _write_progress(results_path, "processing", 0, total, [])

    # Phase 1: Parse all resumes
    processed_resumes = []
    for idx, file_path in enumerate(file_paths):
        try:
            resume_text = extract_text(file_path)
            if not resume_text.strip():
                _write_progress(results_path, "processing", idx + 1, total, [],
                                note=f"Skipped (empty): {os.path.basename(file_path)}")
                continue

            emails, names = extract_entities(resume_text)
            experience = extract_experience(resume_text)
            filename = os.path.basename(file_path)
            processed_resumes.append((names, emails, resume_text, filename, experience))
        except Exception as e:
            _write_progress(results_path, "processing", idx + 1, total, [],
                            note=f"Error on {os.path.basename(file_path)}: {str(e)}")
            continue

        # Update progress after each file
        _write_progress(results_path, "processing", idx + 1, total, [])

    if not processed_resumes:
        _write_progress(results_path, "completed", total, total, [],
                        note="No valid resumes found.")
        return

    # Phase 2: TF-IDF Ranking
    _write_progress(results_path, "ranking", total, total, [],
                    note="Computing TF-IDF similarity scores...")

    tfidf_vectorizer = TfidfVectorizer()
    job_desc_vector = tfidf_vectorizer.fit_transform([job_description])

    ranked_resumes = []
    for (names, emails, resume_text, filename, experience) in processed_resumes:
        resume_vector = tfidf_vectorizer.transform([resume_text])
        base_similarity = cosine_similarity(job_desc_vector, resume_vector)[0][0] * 100
        
        # Calculate skill weight bonus
        if skill_weights:
            total_possible_weight = sum(skill_weights.values())
            matched_weight = 0
            resume_text_lower = resume_text.lower()
            for skill, weight in skill_weights.items():
                if skill.lower() in resume_text_lower:
                    matched_weight += weight
            
            if total_possible_weight > 0:
                skill_match_pct = (matched_weight / total_possible_weight) * 100
                similarity = (base_similarity * 0.5) + (skill_match_pct * 0.5)
            else:
                similarity = base_similarity
        else:
            similarity = base_similarity

        ranked_resumes.append({
            "names": names,
            "emails": emails,
            "similarity": round(similarity, 2),
            "filename": filename,
            "experience": experience
        })

    if ranking_preference == 'experience':
        # Primary sort: similarity descending; tiebreaker: experience descending
        ranked_resumes.sort(key=lambda x: (x["similarity"], x["experience"]), reverse=True)
    else:
        # Sort resumes by similarity score only
        ranked_resumes.sort(key=lambda x: x["similarity"], reverse=True)

    # Phase 3: Done
    _write_progress(results_path, "completed", total, total, ranked_resumes)


def _write_progress(results_path, status, processed, total, results, note=""):
    """Write a JSON progress file atomically."""
    data = {
        "status": status,
        "processed": processed,
        "total": total,
        "results": results,
        "note": note,
    }
    # Write to temp file first, then rename for atomicity
    tmp_path = results_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    # On Windows, os.replace handles atomic rename
    os.replace(tmp_path, results_path)
