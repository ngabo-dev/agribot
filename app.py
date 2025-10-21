"""
Rwanda Smart Farmer Chatbot - Gradio Web Application
====================================================

This file provides a standalone web application for the Rwanda Smart Farmer Chatbot.
Run this file to launch the chatbot interface without running the full notebook.

Usage:
    python app.py

Requirements:
    - Trained model saved in 'models/baseline_model' or 'models/best_model'
    - All dependencies from requirements.txt installed
"""

import os
import gradio as gr
import tensorflow as tf
from transformers import T5Tokenizer, TFT5ForConditionalGeneration
import warnings
warnings.filterwarnings('ignore')

# Configuration
MODEL_PATH = "models/baseline_model"  # Change to your best model path
MAX_INPUT_LENGTH = 128
MAX_OUTPUT_LENGTH = 256

print("="*80)
print("Rwanda Smart Farmer Chatbot 🌾🇷🇼")
print("="*80)

# Load model and tokenizer
print("\nLoading model and tokenizer...")
try:
    if not os.path.exists(MODEL_PATH):
        print(f"⚠️ Model not found at {MODEL_PATH}")
        print("Please train the model first by running the notebook.")
        print("Using pre-trained t5-small as fallback...")
        MODEL_PATH = "t5-small"
    
    tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
    model = TFT5ForConditionalGeneration.from_pretrained(MODEL_PATH)
    print("✅ Model and tokenizer loaded successfully!")
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("Loading fallback model (t5-small)...")
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = TFT5ForConditionalGeneration.from_pretrained("t5-small")
    print("⚠️ Using untrained model. Results may not be optimal.")

def generate_answer(question):
    """
    Generate answer for a given question.
    
    Args:
        question: Input question string
        
    Returns:
        str: Generated answer
    """
    try:
        # Format input
        input_text = f"question: {question}"
        
        # Tokenize
        inputs = tokenizer(
            input_text,
            max_length=MAX_INPUT_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='tf'
        )
        
        # Generate
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=MAX_OUTPUT_LENGTH,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=2,
            temperature=0.8
        )
        
        # Decode
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer
    
    except Exception as e:
        return f"Sorry, I encountered an error: {str(e)}"

def chatbot_interface(message, history):
    """
    Gradio chatbot interface function.
    
    Args:
        message: User's input message
        history: Conversation history
        
    Returns:
        str: Chatbot's response
    """
    if not message or message.strip() == "":
        return "Please ask me a question about agriculture!"
    
    response = generate_answer(message.strip())
    return response

# Create custom CSS for better styling
custom_css = """
#chatbot {
    height: 600px;
}
.message-row {
    padding: 10px;
}
"""

# Create Gradio interface
demo = gr.ChatInterface(
    fn=chatbot_interface,
    title="🌾 Rwanda Smart Farmer Chatbot 🇷🇼",
    description="""
    <div style='text-align: center; padding: 20px;'>
        <h2>Welcome to Rwanda Smart Farmer Chatbot!</h2>
        <p>I'm your AI assistant for agricultural questions in Rwanda.</p>
        <p>Ask me about crops, pests, fertilizers, planting schedules, and farming best practices.</p>
    </div>
    
    <div style='background-color: #f0f8ff; padding: 15px; border-radius: 10px; margin: 10px 0;'>
        <b>📚 I can help with:</b>
        <ul>
            <li>🌱 Crop management and planting schedules</li>
            <li>🐛 Pest control and prevention methods</li>
            <li>💧 Irrigation and water management</li>
            <li>🌾 Fertilizer recommendations</li>
            <li>🦠 Disease identification and treatment</li>
            <li>📦 Post-harvest handling and storage</li>
            <li>🌍 Soil health and conservation</li>
        </ul>
    </div>
    """,
    examples=[
        "How can I prevent maize stem borer?",
        "What fertilizer should I use for tomatoes?",
        "When should I plant beans in Rwanda?",
        "How do I treat banana bacterial wilt?",
        "What is the best way to store potatoes?",
        "How can I improve soil fertility naturally?",
        "What are the signs of cassava mosaic disease?",
        "How often should I water my vegetable garden?",
        "What is crop rotation and why is it important?",
        "How do I prepare soil for planting coffee?"
    ],
    theme=gr.themes.Soft(
        primary_hue="green",
        secondary_hue="emerald",
    ),
    css=custom_css,
    retry_btn="🔄 Retry",
    undo_btn="↩️ Undo",
    clear_btn="🗑️ Clear Chat",
    submit_btn="Send 📤",
    chatbot=gr.Chatbot(
        height=500,
        show_label=False,
        avatar_images=(
            None,  # User avatar (None = default)
            "https://em-content.zobj.net/thumbs/120/apple/354/sheaf-of-rice_1f33e.png"  # Bot avatar
        )
    )
)

if __name__ == "__main__":
    print("\n" + "="*80)
    print("LAUNCHING GRADIO INTERFACE")
    print("="*80)
    print("\nThe chatbot interface will open in your browser.")
    print("\nConfiguration:")
    print(f"  - Model: {MODEL_PATH}")
    print(f"  - Max input length: {MAX_INPUT_LENGTH}")
    print(f"  - Max output length: {MAX_OUTPUT_LENGTH}")
    print(f"  - Server: http://localhost:7860")
    print("\nPress Ctrl+C to stop the server.")
    print("="*80 + "\n")
    
    # Launch the interface
    demo.launch(
        share=False,  # Set to True to create a public link
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860,
        show_error=True,
        show_api=False
    )
