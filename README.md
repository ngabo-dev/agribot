# Rwanda Smart Farmer Chatbot 🌾🇷🇼

## Project Overview

The Rwanda Smart Farmer Chatbot is an AI-powered conversational assistant designed to help farmers in Rwanda with agricultural queries. It provides answers about crops, pests, fertilizers, weather conditions, and general farming practices.

### Purpose and Relevance

Agriculture is the backbone of Rwanda's economy, with over 70% of the population engaged in farming. Many smallholder farmers lack immediate access to expert agricultural advice. This chatbot bridges that gap by providing:

- **24/7 accessibility** to agricultural information
- **Instant responses** to common farming questions
- **Local context** relevant to Rwandan agriculture
- **Multilingual potential** for broader reach

## Dataset

**Source**: [rajathkumar846/agriculture_faq_qa](https://huggingface.co/datasets/rajathkumar846/agriculture_faq_qa)

The dataset contains question-answer pairs covering:
- Crop diseases and prevention
- Pest management
- Fertilizer recommendations
- Planting schedules
- General agricultural practices

**Dataset Statistics**:
- Total Q&A pairs: ~1000+ (will be updated after loading)
- Domain: Agriculture
- Format: Question-Answer pairs

## Model Architecture

**Base Model**: T5 (Text-to-Text Transfer Transformer)
- **Why T5**: Excellent for generative QA tasks, handles various text-to-text problems
- **Framework**: TensorFlow/Keras with Hugging Face Transformers
- **Approach**: Generative QA (generates free-text answers)

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd agribot

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Linux/Mac
# or
venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
```

## Dependencies

```
transformers>=4.30.0
tensorflow>=2.13.0
datasets>=2.14.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
nltk>=3.8.0
gradio>=3.40.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

## Project Structure

```
agribot/
│
├── README.md
├── requirements.txt
├── rwanda_farmer_chatbot.ipynb    # Main notebook
├── app.py                          # Gradio web interface
├── models/                         # Saved models
│   └── best_model/
├── data/                           # Dataset files
│   └── agriculture_faq.csv
├── results/                        # Training results and metrics
│   └── experiment_results.csv
└── demo_video.mp4                  # Demo video (to be created)
```

## Usage

### 1. Training the Model

Open and run `rwanda_farmer_chatbot.ipynb` to:
1. Load and preprocess the dataset
2. Fine-tune the T5 model
3. Evaluate performance
4. Save the trained model

### 2. Running the Chatbot

#### Option A: Jupyter Notebook
Run the final cells in `rwanda_farmer_chatbot.ipynb` for interactive testing.

#### Option B: Web Interface (Gradio)
```bash
python app.py
```
Then open your browser to `http://localhost:7860`

#### Option C: Command Line
```python
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load model
tokenizer = T5Tokenizer.from_pretrained("./models/best_model")
model = T5ForConditionalGeneration.from_pretrained("./models/best_model")

# Ask question
question = "How can I prevent maize stem borer?"
input_text = f"question: {question}"
inputs = tokenizer(input_text, return_tensors="tf", max_length=512, truncation=True)
outputs = model.generate(**inputs, max_length=150)
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(answer)
```

## Performance Metrics

### Quantitative Metrics
- **BLEU Score**: TBD after training
- **ROUGE-L**: TBD after training
- **Perplexity**: TBD after training
- **F1 Score**: TBD after training

### Hyperparameter Tuning Results

| Experiment | Learning Rate | Batch Size | Epochs | BLEU Score | ROUGE-L | Notes |
|------------|--------------|------------|--------|------------|---------|-------|
| Baseline   | 5e-5         | 8          | 3      | TBD        | TBD     | Initial |
| Exp 1      | 3e-5         | 8          | 3      | TBD        | TBD     | Lower LR |
| Exp 2      | 5e-5         | 16         | 3      | TBD        | TBD     | Larger batch |
| Exp 3      | 5e-5         | 8          | 5      | TBD        | TBD     | More epochs |
| Best       | TBD          | TBD        | TBD    | TBD        | TBD     | Best config |

## Example Conversations

### Example 1: Pest Control
**User**: "How can I prevent maize stem borer?"

**Bot**: "To prevent maize stem borer: 1) Practice crop rotation, 2) Plant early to avoid peak pest season, 3) Use resistant maize varieties, 4) Apply biological control like Trichogramma wasps, 5) Remove and destroy crop residues after harvest."

### Example 2: Fertilizer Advice
**User**: "What fertilizer should I use for tomatoes?"

**Bot**: "For tomatoes, use: 1) NPK 10-20-10 at planting, 2) Side-dress with nitrogen-rich fertilizer during growth, 3) Apply calcium to prevent blossom end rot, 4) Use organic compost to improve soil health."

### Example 3: Planting Schedule
**User**: "When should I plant beans in Rwanda?"

**Bot**: "In Rwanda, plant beans during: 1) Season A (September-November) for long rains, 2) Season B (February-April) for short rains. Choose varieties suited to your altitude and local conditions."

## Key Features

✅ Domain-specific agriculture knowledge
✅ Generative QA using T5 transformer
✅ Comprehensive preprocessing pipeline
✅ Multiple hyperparameter experiments
✅ Thorough evaluation metrics
✅ User-friendly Gradio interface
✅ Well-documented codebase

## Future Enhancements

- 🌍 Multilingual support (English, Kinyarwanda, French)
- 📱 Mobile app integration
- 🌤️ Real-time weather integration
- 📍 Location-based recommendations
- 🖼️ Image-based disease detection
- 💬 Voice input/output capabilities

## Contributors

- Your Name - Initial development

## License

This project is licensed under the MIT License.

## Acknowledgments

- Dataset: [rajathkumar846/agriculture_faq_qa](https://huggingface.co/datasets/rajathkumar846/agriculture_faq_qa)
- Hugging Face Transformers library
- TensorFlow team
- Rwanda Agriculture Board for domain expertise

## Contact

For questions or collaboration:
- Email: j.niyongabo@alustudent.com
- GitHub: ngabodev

---

**Note**: This chatbot is designed for educational and informational purposes. For critical agricultural decisions, please consult with local agricultural extension officers or experts.
