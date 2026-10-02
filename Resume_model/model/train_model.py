"""
train_model.py — ResumeIQ
Fine-tunes the embedding model on JD-Resume pairs.
"""
import json
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# 1. Load the pre-trained base model
model_name = "./fine_tuned_resume_model"
model = SentenceTransformer(model_name)

# 2. Load your prepared data
with open(r"c:\Resume_model\training_data.json", "r") as f:
    raw_data = json.load(f)

train_examples = [
    InputExample(texts=item["texts"], label=float(item["label"])) 
    for item in raw_data
]

# 3. Setup DataLoader and Loss
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.CosineSimilarityLoss(model)

# 4. Train (Fine-tune)
print(f"Starting fine-tuning on {len(train_examples)} pairs...")
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=4,
    warmup_steps=100,
    output_path="./fine_tuned_resume_model_v2"
)
print("Training complete! Model saved to ./fine_tuned_resume_model_v2")