#!/usr/bin/env python3
"""
Upload trained model to Hugging Face Hub
This is the recommended way to share large ML models
"""

from huggingface_hub import HfApi, create_repo, login
from transformers import T5ForConditionalGeneration, T5Tokenizer
import os
import sys

# Configuration
MODEL_PATH = "./models/baseline_final"
HF_USERNAME = "ngabodevv"  # Your Hugging Face username (update if different)
MODEL_NAME = "rwanda-farmer-chatbot-t5"
REPO_ID = f"{HF_USERNAME}/{MODEL_NAME}"

def upload_to_huggingface():
    """Upload model to Hugging Face Hub"""
    
    print("="*80)
    print("UPLOADING MODEL TO HUGGING FACE HUB")
    print("="*80)
    
    # Check if logged in
    try:
        api = HfApi()
        user_info = api.whoami()
        print(f"\n👤 Logged in as: {user_info['name']}")
    except Exception as e:
        print("\n❌ Not logged in to Hugging Face!")
        print("💡 Run: huggingface-cli login")
        sys.exit(1)
    
    # 1. Create repository
    print(f"\n📦 Creating repository: {REPO_ID}")
    try:
        create_repo(REPO_ID, exist_ok=True, repo_type="model")
        print("✅ Repository created/exists")
    except Exception as e:
        print(f"❌ Error creating repository: {e}")
        print("💡 Make sure you have write access and correct username")
        sys.exit(1)
    
    # 2-3. Upload entire folder to hub (no local load required)
    print(f"\n⬆️  Uploading folder {MODEL_PATH} to Hugging Face Hub...")
    print("   This may take several minutes depending on your connection...")
    try:
        api = HfApi()
        api.upload_folder(
            repo_id=REPO_ID,
            folder_path=MODEL_PATH,
            path_in_repo=".",
            commit_message="Add model (weights, config, tokenizer)",
        )
        print("✅ Folder uploaded")
    except Exception as e:
        print(f"❌ Error uploading: {e}")
        sys.exit(1)
    
    print("\n" + "="*80)
    print("✅ MODEL UPLOADED SUCCESSFULLY!")
    print("="*80)
    print(f"\n🔗 Your model is now available at:")
    print(f"   https://huggingface.co/{REPO_ID}")
    print(f"\n💡 To use your model:")
    print(f"   from transformers import T5ForConditionalGeneration, T5Tokenizer")
    print(f"   model = T5ForConditionalGeneration.from_pretrained('{REPO_ID}')")
    print(f"   tokenizer = T5Tokenizer.from_pretrained('{REPO_ID}')")

if __name__ == "__main__":
    print("="*80)
    print("🤗 HUGGING FACE MODEL UPLOAD SCRIPT")
    print("="*80)
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"\n❌ Model not found at {MODEL_PATH}")
        print("\n💡 Options:")
        print("   1. Train your model first: run rwanda_farmer_chatbot.ipynb")
        print("   2. Update MODEL_PATH in this script if model is elsewhere")
        sys.exit(1)
    
    print(f"\n📁 Model found at: {MODEL_PATH}")
    print(f"📦 Target repository: {REPO_ID}")
    print(f"\n⚠️  Make sure you've logged in first:")
    print(f"   huggingface-cli login")
    
    input("\n✅ Press Enter to start upload (or Ctrl+C to cancel)...")
    
    upload_to_huggingface()
