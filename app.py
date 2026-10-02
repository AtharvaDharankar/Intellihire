from flask import Flask, render_template, request, redirect, url_for, send_file, session, jsonify
from datetime import datetime
import spacy
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import os
import json
import uuid
import time
import zipfile

app = Flask(__name__)
app.secret_key = 'resume-ranker-secret-key'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB per chunk request
app.config['MAX_FORM_MEMORY_SIZE'] = 50 * 1024 * 1024

UPLOAD_FOLDER = os.path.join(os.path.abspath(os.path.dirname(__file__)), "uploads")
JOBS_FOLDER = os.path.join(os.path.abspath(os.path.dirname(__file__)), "jobs")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(JOBS_FOLDER, exist_ok=True)

nlp = spacy.load("en_core_web_sm")

results = []


USE_RQ = False
redis_conn = None
task_queue = None

try:
    from redis import Redis
    from rq import Queue
    redis_conn = Redis(host='localhost', port=6379, db=0)
    redis_conn.ping()  
    task_queue = Queue(connection=redis_conn, default_timeout=600)
    USE_RQ = True
    print(" * RQ worker mode ENABLED (Redis is running)")
except Exception as e:
    print(f" * RQ worker mode DISABLED (Redis not available: {e})")
    print(" * Falling back to synchronous in-process analysis")



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



def get_upload_session_dir(upload_id):
    """Get directory for a specific chunked-upload session."""
    return os.path.join(UPLOAD_FOLDER, f"session_{upload_id}")


def get_chunk_dir(upload_id, filename):
    """Get chunk storage directory for a specific file in a session."""
    safe_name = re.sub(r'[^\w\-.]', '_', filename)
    return os.path.join(get_upload_session_dir(upload_id), f"chunks_{safe_name}")


def reassemble_file(upload_id, filename, total_chunks):
    """Reassemble chunked file into a single file."""
    chunk_dir = get_chunk_dir(upload_id, filename)
    session_dir = get_upload_session_dir(upload_id)
    safe_name = re.sub(r'[^\w\-.]', '_', filename)
    output_path = os.path.join(session_dir, safe_name)

    with open(output_path, 'wb') as output_file:
        for i in range(total_chunks):
            chunk_path = os.path.join(chunk_dir, f"chunk_{i}")
            if not os.path.exists(chunk_path):
                raise FileNotFoundError(f"Missing chunk {i} for {filename}")
            with open(chunk_path, 'rb') as chunk_file:
                output_file.write(chunk_file.read())


    import shutil
    shutil.rmtree(chunk_dir, ignore_errors=True)

    return output_path


SUPPORTED_RESUME_EXTS = {'.pdf', '.docx', '.txt'}


def extract_zip(zip_path, extract_to):
    """
    Extract a ZIP file and return paths to all supported resume files inside.
    Skips OS-generated metadata folders like __MACOSX.
    """
    extracted_paths = []
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            for member in zf.namelist():
                # Skip directories, hidden files, and macOS metadata
                if member.endswith('/') or '/__MACOSX/' in member or member.startswith('__MACOSX'):
                    continue
                basename = os.path.basename(member)
                if not basename or basename.startswith('.'):
                    continue
                ext = os.path.splitext(basename)[1].lower()
                if ext not in SUPPORTED_RESUME_EXTS:
                    continue
                # Extract to a flat directory (avoid nested folder issues)
                safe_name = re.sub(r'[^\w\-.]', '_', basename)
                target_path = os.path.join(extract_to, safe_name)
                # Handle duplicate names inside the zip
                counter = 1
                while os.path.exists(target_path):
                    name_part, ext_part = os.path.splitext(safe_name)
                    target_path = os.path.join(extract_to, f"{name_part}_{counter}{ext_part}")
                    counter += 1
                with zf.open(member) as src, open(target_path, 'wb') as dst:
                    dst.write(src.read())
                extracted_paths.append(target_path)
    except zipfile.BadZipFile:
        pass  # Skip corrupt zip files silently
    return extracted_paths


# --- Routes ---

@app.route('/', methods=['GET'])
def index():
    """Upload page — the landing page."""
    return render_template('upload.html')


# ---------- Chunked Upload API ----------

@app.route('/api/upload/init', methods=['POST'])
def upload_init():
    """Initialize a new upload session. Returns a unique upload_id."""
    upload_id = str(uuid.uuid4())
    session_dir = get_upload_session_dir(upload_id)
    os.makedirs(session_dir, exist_ok=True)

    # Save metadata
    meta = {
        "upload_id": upload_id,
        "files": {},
        "created_at": time.time()
    }
    with open(os.path.join(session_dir, "meta.json"), "w") as f:
        json.dump(meta, f)

    return jsonify({"upload_id": upload_id})


@app.route('/api/upload/chunk', methods=['POST'])
def upload_chunk():
    """
    Receive a single chunk of a file.
    Expects multipart form data:
      - upload_id: the session id
      - filename: original filename
      - chunk_index: 0-based chunk number
      - total_chunks: total number of chunks for this file
      - chunk: the binary chunk data
    """
    upload_id = request.form.get('upload_id')
    filename = request.form.get('filename')
    chunk_index = int(request.form.get('chunk_index', 0))
    total_chunks = int(request.form.get('total_chunks', 1))
    chunk = request.files.get('chunk')

    if not all([upload_id, filename, chunk]):
        return jsonify({"error": "Missing required fields"}), 400

    session_dir = get_upload_session_dir(upload_id)
    if not os.path.exists(session_dir):
        return jsonify({"error": "Invalid upload session"}), 404

    # Save chunk
    chunk_dir = get_chunk_dir(upload_id, filename)
    os.makedirs(chunk_dir, exist_ok=True)
    chunk_path = os.path.join(chunk_dir, f"chunk_{chunk_index}")
    chunk.save(chunk_path)

    # Update metadata
    meta_path = os.path.join(session_dir, "meta.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)

    if filename not in meta["files"]:
        meta["files"][filename] = {
            "total_chunks": total_chunks,
            "received_chunks": []
        }
    if chunk_index not in meta["files"][filename]["received_chunks"]:
        meta["files"][filename]["received_chunks"].append(chunk_index)
    meta["files"][filename]["received_chunks"].sort()

    with open(meta_path, "w") as f:
        json.dump(meta, f)

    received = len(meta["files"][filename]["received_chunks"])
    is_complete = received == total_chunks

    return jsonify({
        "filename": filename,
        "chunk_index": chunk_index,
        "received": received,
        "total_chunks": total_chunks,
        "file_complete": is_complete
    })


@app.route('/api/upload/finalize', methods=['POST'])
def upload_finalize():
    """
    Reassemble all chunked files and kick off the analysis.
    Expects JSON body: { upload_id, job_description }
    """
    data = request.get_json()
    upload_id = data.get('upload_id')
    job_description = data.get('job_description', '')
    ranking_preference = data.get('ranking_preference', 'skillset')
    skill_weights = data.get('skill_weights', {})

    if not upload_id or not job_description.strip():
        return jsonify({"error": "Missing upload_id or job_description"}), 400

    session_dir = get_upload_session_dir(upload_id)
    meta_path = os.path.join(session_dir, "meta.json")

    if not os.path.exists(meta_path):
        return jsonify({"error": "Invalid upload session"}), 404

    with open(meta_path, "r") as f:
        meta = json.load(f)

    # Reassemble all files
    reassembled_paths = []
    for filename, info in meta["files"].items():
        try:
            path = reassemble_file(upload_id, filename, info["total_chunks"])
            reassembled_paths.append(path)
        except FileNotFoundError as e:
            return jsonify({"error": str(e)}), 400

    if not reassembled_paths:
        return jsonify({"error": "No files uploaded"}), 400

    # Expand ZIP archives into individual resume files
    file_paths = []
    for path in reassembled_paths:
        if os.path.splitext(path)[1].lower() == '.zip':
            extracted = extract_zip(path, session_dir)
            file_paths.extend(extracted)
            # Remove the zip after extraction
            try:
                os.remove(path)
            except OSError:
                pass
        else:
            file_paths.append(path)

    if not file_paths:
        return jsonify({"error": "No supported resume files found (PDF, DOCX, TXT)"}), 400

    # Create a job_id for tracking
    job_id = str(uuid.uuid4())
    results_path = os.path.join(JOBS_FOLDER, f"{job_id}.json")

    if USE_RQ:
        # --- Async path: Enqueue to RQ ---
        from tasks import analyze_resumes_task
        task_queue.enqueue(
            analyze_resumes_task,
            job_description,
            file_paths,
            results_path,
            ranking_preference,
            skill_weights,
            job_id=job_id,
            result_ttl=3600
        )
    else:
        # --- Sync path: Run immediately (blocks request, but good for local/testing) ---
        _run_analysis_sync(job_description, file_paths, results_path, ranking_preference, skill_weights)

    return jsonify({"job_id": job_id})


@app.route('/api/job/<job_id>/status', methods=['GET'])
def job_status(job_id):
    """Poll the status of an analysis job."""
    results_path = os.path.join(JOBS_FOLDER, f"{job_id}.json")

    if not os.path.exists(results_path):
        return jsonify({"status": "queued", "processed": 0, "total": 0, "results": []})

    try:
        with open(results_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return jsonify(data)
    except (json.JSONDecodeError, IOError):
        return jsonify({"status": "queued", "processed": 0, "total": 0, "results": []})


@app.route('/api/job/<job_id>/complete', methods=['POST'])
def job_complete(job_id):
    """
    Called by the frontend once the user acknowledges results.
    Stores results in the global variable for the results page and CSV.
    """
    global results
    results_path = os.path.join(JOBS_FOLDER, f"{job_id}.json")

    if not os.path.exists(results_path):
        return jsonify({"error": "Job not found"}), 404

    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if data.get("status") != "completed":
        return jsonify({"error": "Job not yet completed"}), 400

    # Convert back to the tuple format used by results.html
    ranked = []
    for r in data.get("results", []):
        ranked.append((
            r.get("names", []),
            r.get("emails", []),
            r.get("similarity", 0),
            r.get("filename", ""),
            r.get("experience", 0)
        ))
    results = ranked
    return jsonify({"ok": True, "redirect": url_for('results_page')})


# --- Synchronous fallback (used when Redis is not available) ---

def _run_analysis_sync(job_description, file_paths, results_path, ranking_preference='skillset', skill_weights=None):
    """Run the analysis synchronously within the Flask process."""
    if skill_weights is None:
        skill_weights = {}
    total = len(file_paths)
    processed_resumes = []

    for idx, file_path in enumerate(file_paths):
        try:
            resume_text = extract_text(file_path)
            if not resume_text.strip():
                continue
            emails, names = extract_entities(resume_text)
            experience = extract_experience(resume_text)
            filename = os.path.basename(file_path)
            processed_resumes.append((names, emails, resume_text, filename, experience))
        except Exception:
            continue

    ranked_resumes = []
    if processed_resumes:
        tfidf_vectorizer = TfidfVectorizer()
        job_desc_vector = tfidf_vectorizer.fit_transform([job_description])

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
            # Experience not considered
            ranked_resumes.sort(key=lambda x: x["similarity"], reverse=True)

    data = {
        "status": "completed",
        "processed": total,
        "total": total,
        "results": ranked_resumes,
        "note": ""
    }
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(data, f)


# --- Legacy /analyze route (kept for backward compatibility) ---

@app.route('/analyze', methods=['POST'])
def analyze():
    """Process uploaded resumes and redirect to results."""
    global results

    job_description = request.form['job_description']
    ranking_preference = request.form.get('ranking_preference', 'skillset')
    resume_files = request.files.getlist('resume_files')

    # Create upload directory if it doesn't exist
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    # Process uploaded resumes
    processed_resumes = []
    for resume_file in resume_files:
        if resume_file.filename == '':
            continue

        # Save the uploaded file
        resume_path = os.path.join(UPLOAD_FOLDER, resume_file.filename)
        resume_file.save(resume_path)

        # Extract text based on file type
        resume_text = extract_text(resume_path)
        if not resume_text.strip():
            continue

        emails, names = extract_entities(resume_text)
        experience = extract_experience(resume_text)
        processed_resumes.append((names, emails, resume_text, resume_file.filename, experience))

    if not processed_resumes:
        return redirect(url_for('index'))

    # TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer()
    job_desc_vector = tfidf_vectorizer.fit_transform([job_description])

    # Rank resumes based on similarity
    ranked_resumes = []
    for (names, emails, resume_text, filename, experience) in processed_resumes:
        resume_vector = tfidf_vectorizer.transform([resume_text])
        similarity = cosine_similarity(job_desc_vector, resume_vector)[0][0] * 100
        ranked_resumes.append((names, emails, similarity, filename, experience))

    if ranking_preference == 'experience':
        # Primary sort: similarity descending; tiebreaker: experience descending
        ranked_resumes.sort(key=lambda x: (x[2], x[4]), reverse=True)
    else:
        # Sort resumes by similarity score only
        ranked_resumes.sort(key=lambda x: x[2], reverse=True)

    results = ranked_resumes

    return redirect(url_for('results_page'))


@app.route('/results')
def results_page():
    """Results page — shows ranked resumes."""
    return render_template('results.html', results=results)


def _find_uploaded_file(filename):
    """Locate an uploaded file in the uploads directory tree."""
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    if os.path.exists(file_path):
        return file_path
    # Search in session subdirectories
    for entry in os.listdir(UPLOAD_FOLDER):
        session_path = os.path.join(UPLOAD_FOLDER, entry)
        if os.path.isdir(session_path):
            candidate = os.path.join(session_path, filename)
            if os.path.exists(candidate):
                return candidate
    return None


@app.route('/view/<filename>')
def view_file(filename):
    """Serve an uploaded file for inline viewing (PDF rendered inline)."""
    file_path = _find_uploaded_file(filename)
    if not file_path:
        return "File not found", 404

    ext = os.path.splitext(filename)[1].lower()

    if ext == ".pdf":
        return send_file(file_path, mimetype='application/pdf')
    elif ext == ".docx":
        return send_file(file_path, as_attachment=True, download_name=filename)
    elif ext == ".txt":
        return send_file(file_path, mimetype='text/plain')
    else:
        return send_file(file_path, as_attachment=True, download_name=filename)


@app.route('/view_content/<filename>')
def view_content(filename):
    """
    Return file content as styled HTML for the in-page document viewer.
    PDF  → redirect to /view/<filename> (handled by iframe directly)
    DOCX → convert paragraphs + tables to HTML
    TXT  → wrap plain text in styled HTML
    """
    file_path = _find_uploaded_file(filename)
    if not file_path:
        return "File not found", 404

    ext = os.path.splitext(filename)[1].lower()

    # Common HTML wrapper
    def wrap_html(body_content):
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
  body {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: #111118; color: #e0e0eb;
    margin: 0; padding: 32px 40px;
    line-height: 1.7; font-size: 14px;
  }}
  h1 {{ font-size: 1.5rem; margin-bottom: 8px; color: #f0f0f5; }}
  h2 {{ font-size: 1.2rem; margin-top: 24px; margin-bottom: 6px; color: #d0d0e0; }}
  h3 {{ font-size: 1rem; margin-top: 20px; margin-bottom: 4px; color: #c0c0d5; }}
  p {{ margin: 6px 0; }}
  table {{ width: 100%; border-collapse: collapse; margin: 16px 0; }}
  th, td {{
    border: 1px solid rgba(255,255,255,0.1);
    padding: 8px 12px; text-align: left; font-size: 0.85rem;
  }}
  th {{ background: rgba(139,92,246,0.1); color: #b8a5f5; font-weight: 600; }}
  td {{ color: #c8c8d8; }}
  pre {{
    background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08);
    border-radius: 8px; padding: 20px; white-space: pre-wrap;
    word-wrap: break-word; font-family: 'Consolas', 'Courier New', monospace;
    font-size: 0.85rem; line-height: 1.6; color: #d0d0e0;
  }}
  .bold {{ font-weight: 700; }}
  .italic {{ font-style: italic; }}
  ul, ol {{ margin: 8px 0; padding-left: 24px; }}
  li {{ margin: 3px 0; }}
</style>
</head>
<body>{body_content}</body>
</html>"""

    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        import html as html_mod
        escaped = html_mod.escape(content)
        return wrap_html(f"<pre>{escaped}</pre>")

    elif ext == ".docx":
        try:
            import docx
            from docx.enum.text import WD_ALIGN_PARAGRAPH
        except ImportError:
            return wrap_html("<p>python-docx is required to view DOCX files.</p>")

        doc = docx.Document(file_path)
        parts = []

        for element in doc.element.body:
            tag = element.tag.split('}')[-1] if '}' in element.tag else element.tag

            if tag == 'p':
                # Find matching paragraph object
                for para in doc.paragraphs:
                    if para._element is element:
                        text = para.text.strip()
                        if not text:
                            continue
                        style_name = para.style.name.lower() if para.style else ''
                        if 'heading 1' in style_name:
                            parts.append(f"<h1>{text}</h1>")
                        elif 'heading 2' in style_name:
                            parts.append(f"<h2>{text}</h2>")
                        elif 'heading 3' in style_name:
                            parts.append(f"<h3>{text}</h3>")
                        else:
                            # Build runs with inline formatting
                            runs_html = []
                            for run in para.runs:
                                rt = run.text
                                if not rt:
                                    continue
                                import html as html_mod
                                rt = html_mod.escape(rt)
                                if run.bold:
                                    rt = f"<strong>{rt}</strong>"
                                if run.italic:
                                    rt = f"<em>{rt}</em>"
                                if run.underline:
                                    rt = f"<u>{rt}</u>"
                                runs_html.append(rt)
                            if runs_html:
                                parts.append(f"<p>{''.join(runs_html)}</p>")
                            elif text:
                                import html as html_mod
                                parts.append(f"<p>{html_mod.escape(text)}</p>")
                        break

            elif tag == 'tbl':
                # Find matching table object
                for table in doc.tables:
                    if table._element is element:
                        rows_html = []
                        for i, row in enumerate(table.rows):
                            cells_html = []
                            cell_tag = "th" if i == 0 else "td"
                            for cell in row.cells:
                                import html as html_mod
                                cells_html.append(
                                    f"<{cell_tag}>{html_mod.escape(cell.text)}</{cell_tag}>"
                                )
                            rows_html.append(f"<tr>{''.join(cells_html)}</tr>")
                        parts.append(f"<table>{''.join(rows_html)}</table>")
                        break

        return wrap_html("\n".join(parts) if parts else "<p>Document is empty.</p>")

    elif ext == ".pdf":
        # PDFs are served directly via /view/<filename> in an iframe
        return redirect(url_for('view_file', filename=filename))

    else:
        return wrap_html("<p>Unsupported file format.</p>")


@app.route('/download_csv')
def download_csv():
    """Generate and download CSV of ranked results."""
    csv_content = "Rank,Name,Email,Similarity,Experience\n"
    for rank, entry in enumerate(results, start=1):
        names, emails, similarity, filename = entry[0], entry[1], entry[2], entry[3]
        experience = entry[4] if len(entry) > 4 else 0
        name = names[0] if names else "N/A"
        email = emails[0] if emails else "N/A"
        exp_label = f"{experience} yrs" if experience else "Fresher"
        csv_content += f"{rank},{name},{email},{similarity},{exp_label}\n"

    csv_filename = "ranked_resumes.csv"
    csv_full_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), csv_filename)
    with open(csv_full_path, "w") as csv_file:
        csv_file.write(csv_content)

    return send_file(csv_full_path, as_attachment=True, download_name="ranked_resumes.csv")


if __name__ == '__main__':
    app.run(debug=True)
