import io
import PyPDF2
import docx

def extract_text_from_file(filename: str, file_bytes: bytes) -> str:
    """
    Extracts text from a given file (pdf, docx, txt) using its bytes.
    """
    ext = filename.lower().split('.')[-1]
    
    try:
        if ext == 'pdf':
            return _extract_from_pdf(file_bytes)
        elif ext in ['doc', 'docx']:
            return _extract_from_docx(file_bytes)
        elif ext == 'txt':
            return file_bytes.decode('utf-8', errors='ignore')
        else:
            return file_bytes.decode('utf-8', errors='ignore')
    except Exception as e:
        return f"Error extracting text from {filename}: {e}"

def _extract_from_pdf(file_bytes: bytes) -> str:
    text = []
    with io.BytesIO(file_bytes) as pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text.append(extracted)
    return "\n".join(text)

def _extract_from_docx(file_bytes: bytes) -> str:
    text = []
    with io.BytesIO(file_bytes) as docx_file:
        doc = docx.Document(docx_file)
        for para in doc.paragraphs:
            text.append(para.text)
    return "\n".join(text)
