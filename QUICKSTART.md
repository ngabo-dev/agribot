# Rwanda Smart Farmer Chatbot - Quick Start Guide

## 📋 Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)
- GPU (optional, but recommended for faster training)

## 🚀 Quick Start

### Step 1: Clone and Setup

```bash
# Navigate to project directory
cd /home/ngabotech/Documents/ALU/ml/agribot

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Linux/Mac
# or
venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Run the Notebook

```bash
# Launch Jupyter
jupyter notebook rwanda_farmer_chatbot.ipynb
```

Or use VS Code to open the notebook directly.

### Step 3: Train the Model

Run all cells in the notebook sequentially:
1. Import libraries
2. Load dataset
3. Preprocess data
4. Train model
5. Evaluate performance
6. Test chatbot

**Note**: Training may take 30-60 minutes depending on your hardware.

### Step 4: Launch Web Interface

After training, you can launch the Gradio interface:

**Option A: From notebook**
- Run the final cell in the notebook

**Option B: Standalone app**
```bash
python app.py
```

The interface will be available at: http://localhost:7860

## 📁 Project Structure

```
agribot/
├── rwanda_farmer_chatbot.ipynb    # Main notebook
├── app.py                          # Standalone Gradio app
├── requirements.txt                # Dependencies
├── README.md                       # Project documentation
├── QUICKSTART.md                   # This file
├── models/                         # Saved models (created during training)
│   └── baseline_model/
├── results/                        # Training results (created during training)
│   └── experiment_results.csv
└── data/                           # Dataset (downloaded automatically)
```

## 🔧 Common Issues and Solutions

### Issue: Out of Memory

**Solution**: Reduce batch size in the notebook
```python
BATCH_SIZE = 4  # Instead of 8
```

### Issue: CUDA/GPU not detected

**Solution**: Install TensorFlow GPU version
```bash
pip install tensorflow[and-cuda]
```

Or continue with CPU (training will be slower).

### Issue: Dataset download fails

**Solution**: Check internet connection or download manually
```python
```python
# Example: Load the dataset
from datasets import load_dataset
dataset = load_dataset("KisanVaani/agriculture-qa-english-only")
```

### Issue: Model loading fails

**Solution**: Retrain the model or use pre-trained T5
```python
model = TFT5ForConditionalGeneration.from_pretrained("t5-small")
```

## 📊 Expected Training Time

| Hardware | Training Time (3 epochs) |
|----------|-------------------------|
| CPU only | 2-3 hours |
| GPU (RTX 3060) | 30-45 minutes |
| GPU (RTX 4090) | 15-20 minutes |

## 🎯 Evaluation Metrics Guide

- **BLEU Score** (0-1): Higher is better
  - > 0.40: Excellent
  - 0.25-0.40: Good
  - 0.10-0.25: Fair
  - < 0.10: Needs improvement

- **ROUGE-L** (0-1): Higher is better
  - > 0.50: Excellent
  - 0.35-0.50: Good
  - 0.20-0.35: Fair
  - < 0.20: Needs improvement

- **F1 Score** (0-1): Higher is better
  - > 0.60: Excellent
  - 0.45-0.60: Good
  - 0.30-0.45: Fair
  - < 0.30: Needs improvement

## 🔄 Hyperparameter Tuning Tips

### To improve performance:

1. **Increase training epochs**
   ```python
   EPOCHS = 5  # or more
   ```

2. **Try different learning rates**
   ```python
   LEARNING_RATE = 3e-5  # Lower for stability
   LEARNING_RATE = 1e-4  # Higher for faster learning
   ```

3. **Adjust batch size**
   ```python
   BATCH_SIZE = 16  # If you have enough memory
   ```

4. **Use larger model**
   ```python
   model_name = "t5-base"  # Instead of t5-small
   ```

## 📱 Deployment Options

### Option 1: Local Gradio (Current)
```bash
python app.py
```

### Option 2: Hugging Face Spaces
1. Create account on huggingface.co
2. Create new Space
3. Upload model and app.py
4. Deploy automatically

### Option 3: Streamlit Cloud
1. Convert to Streamlit app
2. Deploy on streamlit.io
3. Share public link

### Option 4: Docker Container
```bash
# Create Dockerfile
docker build -t rwanda-farmer-bot .
docker run -p 7860:7860 rwanda-farmer-bot
```

## 📹 Demo Video Guidelines

Your 5-10 minute demo should include:

1. **Introduction** (30 sec)
   - Project overview
   - Problem statement

2. **Dataset & Preprocessing** (1-2 min)
   - Data exploration
   - Cleaning steps

3. **Model Training** (2-3 min)
   - Architecture explanation
   - Training process
   - Hyperparameter experiments

4. **Evaluation** (1-2 min)
   - Metrics explanation
   - Results analysis

5. **Live Demo** (2-3 min)
   - Interactive chatbot
   - Sample questions
   - Response quality

6. **Conclusion** (30 sec)
   - Key takeaways
   - Future work

## 🆘 Getting Help

If you encounter issues:

1. Check error messages carefully
2. Review the notebook comments
3. Consult the README.md
4. Check Hugging Face documentation
5. Review TensorFlow/Keras guides

## 📚 Additional Resources

- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [T5 Paper](https://arxiv.org/abs/1910.10683)
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
- [Gradio Documentation](https://gradio.app/docs)

## ✅ Submission Checklist

- [ ] Complete Jupyter notebook with all cells executed
- [ ] README.md with project documentation
- [ ] requirements.txt with all dependencies
- [ ] Trained model saved in models/ directory
- [ ] Results and metrics documented
- [ ] Demo video (5-10 minutes)
- [ ] GitHub repository with clean commit history
- [ ] Example conversations documented
- [ ] Code is well-commented and clean

---

**Good luck with your project! 🌾🇷🇼**
