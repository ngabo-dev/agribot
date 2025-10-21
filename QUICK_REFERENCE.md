# 🚀 QUICK REFERENCE CARD
## Rwanda Smart Farmer Chatbot

---

## 📦 Installation (One Command)

```bash
pip install transformers tensorflow datasets evaluate rouge-score nltk gradio pandas numpy scikit-learn matplotlib seaborn sentencepiece -q
```

---

## 🎯 Running the Project

### Option 1: Jupyter Notebook
```bash
jupyter notebook rwanda_farmer_chatbot.ipynb
# Run all cells sequentially (Shift+Enter)
```

### Option 2: Standalone App
```bash
python app.py
# Open browser to http://localhost:7860
```

---

## 📊 Notebook Structure (29 Cells)

| Section | Cells | What It Does |
|---------|-------|--------------|
| 1. Introduction | 1 | Project overview |
| 2. Libraries | 2-3 | Import dependencies |
| 3. Data Loading | 4-6 | Load & explore dataset |
| 4. Preprocessing | 7-8 | Clean & normalize data |
| 5. Tokenization | 9-10 | Prepare for T5 |
| 6. Data Split | 11-12 | Train/Val/Test split |
| 7. TF Datasets | 13-14 | Create TF datasets |
| 8. Model Setup | 15-16 | Load T5 model |
| 9. Training | 17-18 | Fine-tune model |
| 10. Evaluation | 19-20 | Metrics & analysis |
| 11. Hyperparameters | 21-22 | Tuning experiments |
| 12. Testing | 23-24 | Sample queries |
| 13. Interface | 25-26 | Chatbot class |
| 14. Deployment | 27-28 | Gradio interface |
| 15. Summary | 29 | Project wrap-up |

---

## 🎓 Key Variables to Remember

```python
# Model
model_name = "t5-small"
MAX_INPUT_LENGTH = 128
MAX_TARGET_LENGTH = 256

# Training
BATCH_SIZE = 8
LEARNING_RATE = 5e-5
EPOCHS = 3

# Data Split
train: 70%, val: 15%, test: 15%
```

---

## 📈 Expected Training Time

| Hardware | Time |
|----------|------|
| CPU | 2-3 hours |
| GPU (3060) | 30-45 min |
| GPU (4090) | 15-20 min |

---

## 💬 Testing the Chatbot

```python
# Method 1: Direct function call
answer = generate_answer("Your question here", model, tokenizer)

# Method 2: Chatbot class
chatbot = RwandaFarmerChatbot(model, tokenizer)
answer = chatbot.ask("Your question here")

# Method 3: Gradio interface
python app.py  # Then use web interface
```

---

## 🐛 Common Issues

| Problem | Solution |
|---------|----------|
| Out of memory | Reduce BATCH_SIZE to 4 |
| Slow training | Use GPU or reduce dataset |
| Import errors | Run: pip install -r requirements.txt |
| Model not found | Check models/ directory exists |

---

## 📊 Evaluation Metrics Targets

| Metric | Good | Excellent |
|--------|------|-----------|
| BLEU | > 0.25 | > 0.40 |
| ROUGE-L | > 0.35 | > 0.50 |
| F1 Score | > 0.40 | > 0.60 |

---

## 🎬 Demo Video Sections

1. Intro (30s)
2. Problem & Dataset (1-2min)
3. Model & Training (2min)
4. Hyperparameters (1-2min)
5. Evaluation (1-2min)
6. Live Demo (2-3min)
7. Conclusion (30s)

**Total: 5-10 minutes**

---

## 📝 Example Questions to Test

✅ "How can I prevent maize stem borer?"
✅ "What fertilizer should I use for tomatoes?"
✅ "When should I plant beans in Rwanda?"
✅ "How do I treat banana bacterial wilt?"
✅ "What is crop rotation?"

---

## 🗂️ Files Created

```
✅ rwanda_farmer_chatbot.ipynb  (Main notebook)
✅ app.py                        (Standalone app)
✅ README.md                     (Documentation)
✅ requirements.txt              (Dependencies)
✅ QUICKSTART.md                 (Setup guide)
✅ DEMO_SCRIPT.md                (Video guide)
✅ PROJECT_SUMMARY.md            (Summary)
✅ .gitignore                    (Git config)
```

---

## 🚀 Deployment Command

```bash
# Local
python app.py

# Public (with sharing)
# Set share=True in demo.launch()

# Docker (if needed)
docker build -t agribot .
docker run -p 7860:7860 agribot
```

---

## ✅ Final Submission Checklist

- [ ] All notebook cells executed
- [ ] Model trained and saved
- [ ] Demo video recorded (5-10 min)
- [ ] GitHub repo created
- [ ] README.md completed
- [ ] Code tested and working
- [ ] Requirements.txt verified

---

## 📧 Repository Structure

```
agribot/
├── rwanda_farmer_chatbot.ipynb ⭐ Main file
├── app.py                       ⭐ Deployment
├── README.md                    ⭐ Documentation
├── requirements.txt             ⭐ Dependencies
├── QUICKSTART.md
├── DEMO_SCRIPT.md
├── PROJECT_SUMMARY.md
├── .gitignore
├── models/
│   └── baseline_model/
├── results/
│   └── experiment_results.csv
└── data/
```

---

## 🎯 Quick Commands Reference

```bash
# Setup
pip install -r requirements.txt

# Run notebook
jupyter notebook rwanda_farmer_chatbot.ipynb

# Launch app
python app.py

# Check GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Create directories
mkdir -p models results data

# Git commands
git init
git add .
git commit -m "Rwanda Smart Farmer Chatbot"
git push origin main
```

---

## 🌟 Project Highlights

✅ T5 Transformer Model
✅ 1000+ Agriculture Q&A pairs
✅ Multiple Evaluation Metrics
✅ Hyperparameter Tuning
✅ Gradio Web Interface
✅ Production Ready
✅ Well Documented

---

**Quick Start**: `jupyter notebook rwanda_farmer_chatbot.ipynb`

**Deploy**: `python app.py`

**Good Luck! 🌾🇷🇼**
