import json
import random

def generate_synthetic_data():
    data = []
    
    # 1. TEMPLATES: Skills and Roles
    tech_skills = ["Python", "FastAPI", "React", "Docker", "PyTorch", "SQL"]
    non_tech_skills = ["Graphic Design", "Sales", "Customer Support", "Cooking"]

    # 2. GENERATE GOOD MATCHES (Label 0.8 - 1.0)
    for _ in range(20):
        skill = random.choice(tech_skills)
        resume = f"Experienced developer specialized in {skill} and system architecture."
        jd = f"We are hiring a Senior Engineer with deep expertise in {skill}."
        data.append({"texts": [resume, jd], "label": round(random.uniform(0.8, 1.0), 2)})

    # 3. GENERATE NEUTRAL MATCHES (Label 0.4 - 0.6)
    for _ in range(15):
        resume = "Junior developer with 1 year experience in Python."
        jd = "Hiring a Lead Architect with 10+ years experience in Cloud Infrastructure."
        data.append({"texts": [resume, jd], "label": round(random.uniform(0.3, 0.5), 2)})

    # 4. GENERATE BAD MATCHES (Label 0.0 - 0.2)
    for _ in range(15):
        resume = f"Professional in {random.choice(non_tech_skills)}."
        jd = f"Seeking an expert in {random.choice(tech_skills)}."
        data.append({"texts": [resume, jd], "label": round(random.uniform(0.0, 0.2), 2)})

    # Save to file
    with open("training_data.json", "w") as f:
        json.dump(data, f, indent=4)
    print(f"Generated {len(data)} training pairs in training_data.json")

if __name__ == "__main__":
    generate_synthetic_data()