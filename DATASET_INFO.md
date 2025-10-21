# Agriculture Dataset Information 🌾

## ✅ Selected Dataset: KisanVaani/agriculture-qa-english-only

### Overview
This is the **primary dataset** for your Rwanda Smart Farmer Chatbot project, replacing the non-existent `rajathkumar846/agriculture_faq_qa` dataset.

### Dataset Details

| Property | Value |
|----------|-------|
| **Name** | KisanVaani/agriculture-qa-english-only |
| **Size** | 22,615 Q&A pairs |
| **Language** | English |
| **Format** | Question-Answer pairs |
| **Source** | [Hugging Face](https://huggingface.co/datasets/KisanVaani/agriculture-qa-english-only) |
| **License** | Open source |

### Coverage Areas

The dataset covers comprehensive agricultural topics including:

1. **Crop Management**
   - Crop rotation importance and techniques
   - Planting schedules and timing
   - Growth stages and requirements

2. **Soil Management**
   - Soil health and fertility
   - Erosion prevention
   - Organic matter management

3. **Pest Control**
   - Common agricultural pests
   - Integrated pest management
   - Disease prevention strategies

4. **Farming Practices**
   - Sustainable farming techniques
   - Water management
   - Fertilizer applications
   - Harvesting methods

5. **General Agriculture Knowledge**
   - Agricultural economics
   - Farm planning
   - Equipment and tools

### Sample Data

**Question**: "why is crop rotation important in farming?"

**Answer**: "This helps to prevent soil erosion and depletion, and can also help to control pests and diseases..."

### Usage in Your Project

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("KisanVaani/agriculture-qa-english-only")

# Access training data
train_data = dataset['train']
print(f"Total examples: {len(train_data)}")

# View first example
print(train_data[0])
# {'question': 'why is crop rotation important in farming?', 
#  'answers': 'This helps to prevent...'}
```

### Data Structure

```python
{
  'question': str,  # The farmer's question
  'answers': str    # The agricultural answer (note: plural 'answers')
}
```

**Important**: The dataset uses `'answers'` (plural) as the column name, not `'answer'`. The notebook has been updated to handle this automatically.

---

## 🔄 Alternative Datasets (Backup Options)

If you need additional data or want to experiment:

### Option 2: sowmya14/agriculture_QA
- **Size**: 999 examples
- **Focus**: Pest and crop management
- **Columns**: 'questions', 'answers'
- **Use case**: Specialized pest control knowledge

### Option 3: PRAKALP-PANDE/PSP-agricultureQnA-1k-unique
- **Size**: 1,000 examples
- **Focus**: Comprehensive agriculture
- **Columns**: 'question', 'answers', 'text'
- **Use case**: Broader agricultural topics

### Option 4: CopyleftCultivars/Natural-Farming-Real-QandA-Conversations-Q1-2024-Update
- **Size**: 1,126 examples
- **Focus**: Natural farming conversations
- **Columns**: 'Question', 'Answer'
- **Use case**: Sustainable/organic farming focus

---

## 🚀 Next Steps

Now that you have a working dataset:

1. ✅ **Dataset Selected** - KisanVaani with 22,615 examples
2. ✅ **Notebook Updated** - Code now loads correct dataset
3. ✅ **Documentation Updated** - README, QUICKSTART, PROJECT_SUMMARY
4. ⏭️ **Next: Open Google Colab** - Train the model with GPU
5. ⏭️ **Then: Evaluate Model** - Run all evaluation metrics
6. ⏭️ **Finally: Deploy** - Launch Gradio interface

### Training Recommendations

With 22,615 examples:
- **Training set**: ~18,000 examples (80%)
- **Validation set**: ~2,300 examples (10%)
- **Test set**: ~2,300 examples (10%)

**Estimated training time**:
- Google Colab (T4 GPU): ~45-60 minutes for 3 epochs
- Local CPU: Not recommended (would take hours)

**Hyperparameter baseline**:
```python
learning_rate = 5e-5
batch_size = 8
epochs = 3
```

---

## 📊 Dataset Quality Insights

### Strengths
✅ Large size (22K+ examples)  
✅ Clean English text  
✅ Broad agricultural coverage  
✅ Good for generative QA with T5  
✅ Actively maintained on Hugging Face  

### Considerations
⚠️ Not Rwanda-specific (general agriculture)  
⚠️ May need minor Rwanda context adaptation  
⚠️ English-only (no Kinyarwanda)  

### Adaptation for Rwanda Context

The model will learn general agriculture principles. For Rwanda-specific deployment:
1. Fine-tune on Rwanda climate conditions
2. Add Rwanda crop varieties in prompts
3. Consider post-processing for local context
4. Future: Add Kinyarwanda translation layer

---

## 🔧 Troubleshooting

### If dataset loading fails:

```python
# Option 1: Force download
dataset = load_dataset("KisanVaani/agriculture-qa-english-only", download_mode="force_redownload")

# Option 2: Use specific revision
dataset = load_dataset("KisanVaani/agriculture-qa-english-only", revision="main")

# Option 3: Check internet connection
import requests
response = requests.get("https://huggingface.co")
print(f"HuggingFace accessible: {response.status_code == 200}")
```

### Column Name Issues

The dataset uses `'answers'` (plural). Your notebook automatically renames it:

```python
# This is already in your notebook (cell after loading data)
if 'answers' in df.columns and 'answer' not in df.columns:
    df = df.rename(columns={'answers': 'answer'})
```

---

## 📚 Additional Resources

- [KisanVaani Dataset Page](https://huggingface.co/datasets/KisanVaani/agriculture-qa-english-only)
- [Hugging Face Datasets Documentation](https://huggingface.co/docs/datasets)
- [T5 Model Card](https://huggingface.co/t5-small)

---

**Status**: ✅ Ready to train!  
**Last Updated**: 2024  
**Verified Working**: Yes (tested successfully)
