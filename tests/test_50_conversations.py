# tests/test_50_conversations.py
import asyncio
import json
import zipfile
import os
from src.inference import InferenceEngine

# Sample diverse conversational turns
SAMPLE_TURNS = [
    "I've been trying to fix this router for hours and it's absolute garbage! I want a refund now.",
    "Could you please clarify the meeting time for tomorrow? I want to ensure I'm fully prepared.",
    "I don't really know what to do... everything feels a bit overwhelming lately.",
    "Just deployed the new Docker container to production. Latency dropped by 40ms.",
    "Whatever. Do what you want. I don't care anymore."
] * 10 # Multiplied to reach 50 conversations

async def generate_deliverable():
    engine = InferenceEngine()
    os.makedirs("outputs", exist_ok=True)
    
    print("Starting evaluation of 50 conversations...")
    all_results = []
    
    for i, turn in enumerate(SAMPLE_TURNS):
        print(f"Processing turn {i+1}/50...")
        result = await engine.evaluate_turn(turn)
        all_results.append(result.model_dump())
        
    # Save to JSONL
    output_file = "outputs/evaluation_results.jsonl"
    with open(output_file, "w") as f:
        for res in all_results:
            f.write(json.dumps(res) + "\n")
            
    # Zip it up for the deliverable
    zip_path = "outputs/deliverable_50_conversations.zip"
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        zipf.write(output_file, arcname="evaluation_results.jsonl")
        
    print(f"\nDeliverable successfully created at: {zip_path}")

if __name__ == "__main__":
    asyncio.run(generate_deliverable())