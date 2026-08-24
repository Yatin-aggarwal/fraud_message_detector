import os
from transformers import pipeline
import pandas as pd
import random
import torch 
from dotenv import load_dotenv
import os


load_dotenv()

print("CUDA available:", torch.cuda.is_available())
os.environ["HF_TOKEN"] = os.getenv()

# Load an open-source text generation model
generator = pipeline("text-generation", model="meta-llama/Llama-3.2-3B-Instruct",device=0 )  # You can use a larger model like 'gpt-neo-1.3B'


SYSTEM_PROMPT = "You produce exactly what the user asks. Never repeat instructions. No explanations."

USER_NORMAL = (
    "Write ONE short, natural chat message that involves money "
    "(paying bills, splitting costs, sending money, salary, shopping, casual mention). "
    "No label, no JSON, no quotes—just the message text."
)

USER_FRAUD = (
    "Write ONE short fraudulent financial chat message. "
    "The scam can be of any type, such as OTP phishing, urgent account verification, fake prize claims, "
    "family emergency scams, or impersonation of banks/government/tax authorities. "
    "Make it realistic and convincing but fraudulent. "
    "No labels, no JSON, no quotes—just the scam message."
)
def generate_financial_chat(label):
    if(label == "scam"):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_FRAUD},
        ]
    else:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_NORMAL},
        ]

    result = generator(messages , max_length=30, return_full_text=False,num_return_sequences=4,top_p=0.92,top_k=50 ,do_sample=True, temperature=0.85,repetition_penalty=1.05)
    print(result)
    return result

# Generate dataset
data = []
for _ in range(20):  # Change 50 to 10_000 for a larger dataset
    label = random.choice(["normal", "scam"])
    messages = generate_financial_chat(label)
    for msg in messages:  
        data.append({"label": label, "message": msg['generated_text']})

# Save to CSV
df = pd.DataFrame(data)
df.to_csv("data.csv", index=False)

print("Synthetic dataset saved to synthetic_financial_chats.csv")
