from sentence_transformers import SentenceTransformer, util
import sys

# Load the newly trained model
model_path = "./fine_tuned_resume_model_v2"
print(f"Loading model from {model_path}...")
try:
    model = SentenceTransformer(model_path)
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

# A sample resume text
resume_text = "I am a skilled Software Engineer with 5 years of experience in Python, FastAPI, and Machine Learning. I have built scalable microservices and led a team of 3 developers."

# Job descriptions to test against
good_jd = "Looking for a Senior Software Engineer with strong Python and FastAPI skills. Experience with Machine Learning is a huge plus."
bad_jd = "We are seeking a Graphic Designer proficient in Adobe Illustrator, Photoshop, and Figma to create marketing materials."
partial_jd = "Hiring a Software Engineer with experience in Python and Ruby on Rails. Must know React for frontend development."

# Compute embeddings
print("\nComputing embeddings...")
resume_emb = model.encode(resume_text, convert_to_tensor=True)
good_jd_emb = model.encode(good_jd, convert_to_tensor=True)
bad_jd_emb = model.encode(bad_jd, convert_to_tensor=True)
partial_jd_emb = model.encode(partial_jd, convert_to_tensor=True)

# Calculate cosine similarities
good_score = util.cos_sim(resume_emb, good_jd_emb).item()
bad_score = util.cos_sim(resume_emb, bad_jd_emb).item()
partial_score = util.cos_sim(resume_emb, partial_jd_emb).item()

print("\n--- Model Verification Results ---")
print(f"Resume: '{resume_text}'\n")
print(f"Good Match JD: '{good_jd}'")
print(f"-> Score: {good_score:.4f} (Expected High)\n")

print(f"Bad Match JD: '{bad_jd}'")
print(f"-> Score: {bad_score:.4f} (Expected Low)\n")

print(f"Partial Match JD: '{partial_jd}'")
print(f"-> Score: {partial_score:.4f} (Expected Medium)\n")
