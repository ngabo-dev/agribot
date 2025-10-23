#!/usr/bin/env python3
"""
Save a valid T5 model (t5-small) into models/baseline_final so that
uploading to Hugging Face can proceed. This is a placeholder if fine-tuned
weights aren't present yet. You can re-upload later after training.
"""
import os
from transformers import T5ForConditionalGeneration, T5Tokenizer

TARGET_DIR = os.path.join('models', 'baseline_final')

def main():
    os.makedirs(TARGET_DIR, exist_ok=True)
    print(f"Saving base t5-small model to: {TARGET_DIR}")

    model = T5ForConditionalGeneration.from_pretrained('t5-small')
    tokenizer = T5Tokenizer.from_pretrained('t5-small')

    model.save_pretrained(TARGET_DIR)
    tokenizer.save_pretrained(TARGET_DIR)

    # Confirm key files
    expected = [
        'pytorch_model.bin',
        'config.json',
        'tokenizer_config.json',
        'special_tokens_map.json',
        'spiece.model'
    ]
    missing = [f for f in expected if not os.path.exists(os.path.join(TARGET_DIR, f))]
    if missing:
        print(f"⚠️ Missing files after save: {missing}")
    else:
        print("✅ All core model/tokenizer files saved!")

if __name__ == '__main__':
    main()
