import os
from pathlib import Path
import PyPDF2

def batch_convert_pdf_to_txt(source_folder, output_folder="resume_data"):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created folder: {output_folder}")

    # Path object for the source folder
    src_path = Path(source_folder)
    
    # Filter for PDF files
    pdf_files = list(src_path.glob("*.pdf"))
    
    if not pdf_files:
        print("No PDF files found in the source folder.")
        return

    print(f"Found {len(pdf_files)} PDFs. Starting conversion...")

    for pdf_file in pdf_files:
        try:
            # Define output filename (same name, .txt extension)
            txt_filename = pdf_file.stem + ".txt"
            txt_path = os.path.join(output_folder, txt_filename)
            
            with open(pdf_file, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                content = []
                
                # Extract text from every page
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        content.append(text)
                
                # Save to the resume_data folder
                with open(txt_path, 'w', encoding='utf-8') as txt_out:
                    txt_out.write("\n".join(content))
            
            print(f"Converted: {pdf_file.name} -> {txt_filename}")
            
        except Exception as e:
            print(f"Failed to convert {pdf_file.name}: {e}")

# --- Execution ---
# Set the folder where your PDFs are currently located
source_directory = "resume_data_pdf" 
batch_convert_pdf_to_txt(source_directory)