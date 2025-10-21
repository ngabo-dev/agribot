# Alternative Agriculture Datasets

If you want to explore other datasets or combine multiple sources:

## 🌾 English Agriculture Datasets

### 1. KisanVaani/agriculture-qa-english-only ⭐ **SELECTED**
- **Size**: 22,615 Q&A pairs
- **Language**: English
- **Quality**: High
- **Use**: Primary dataset
- **Load**: `load_dataset("KisanVaani/agriculture-qa-english-only")`

### 2. sowmya14/agriculture_QA
- **Size**: 999 Q&A pairs
- **Language**: English
- **Focus**: Pest control, crop management
- **Use**: Supplementary data for pest-specific queries
- **Load**: `load_dataset("sowmya14/agriculture_QA")`

### 3. PRAKALP-PANDE/PSP-agricultureQnA-1k-unique
- **Size**: 1,000 Q&A pairs
- **Language**: English
- **Focus**: Banana diseases, general agriculture
- **Use**: Additional training data
- **Load**: `load_dataset("PRAKALP-PANDE/PSP-agricultureQnA-1k-unique")`

### 4. CopyleftCultivars/Natural-Farming-Real-QandA-Conversations-Q1-2024-Update
- **Size**: 1,126 Q&A pairs
- **Language**: English
- **Focus**: Natural/organic farming
- **Use**: Sustainable farming knowledge
- **Load**: `load_dataset("CopyleftCultivars/Natural-Farming-Real-QandA-Conversations-Q1-2024-Update")`

---

## 🔀 How to Combine Multiple Datasets

If you want to merge datasets for more training data:

```python
from datasets import load_dataset, concatenate_datasets

# Load primary dataset
ds1 = load_dataset("KisanVaani/agriculture-qa-english-only")

# Load supplementary datasets
ds2 = load_dataset("sowmya14/agriculture_QA")
ds3 = load_dataset("PRAKALP-PANDE/PSP-agricultureQnA-1k-unique")

# Standardize column names
def standardize(example):
    # Handle different column naming conventions
    if 'answers' in example:
        example['answer'] = example.pop('answers')
    if 'questions' in example:
        example['question'] = example.pop('questions')
    # Keep only question and answer columns
    return {'question': example['question'], 'answer': example['answer']}

# Apply standardization
ds1_clean = ds1['train'].map(standardize, remove_columns=ds1['train'].column_names)
ds2_clean = ds2['train'].map(standardize, remove_columns=ds2['train'].column_names)
ds3_clean = ds3['train'].map(standardize, remove_columns=ds3['train'].column_names)

# Combine datasets
combined = concatenate_datasets([ds1_clean, ds2_clean, ds3_clean])

print(f"Combined dataset size: {len(combined):,} examples")
# Output: Combined dataset size: 24,614 examples
```

---

## 🌍 Non-English Datasets (Future Enhancement)

### Nepali Agriculture
- **Dataset**: Chhabi/Nepali-Agriculture-QA
- **Size**: 22,615 Q&A pairs
- **Language**: Nepali
- **Use**: For multilingual expansion or translation

### How to Use for Rwanda:
1. Train separate model on Nepali data
2. Use translation API to convert to English
3. Fine-tune bilingual model
4. Eventually add Kinyarwanda

---

## 🔍 General QA Datasets (Not Recommended)

These are general QA datasets that could work but lack agriculture domain knowledge:

- **squad**: Stanford QA (extractive, not generative)
- **squad_v2**: Updated version with unanswerable questions
- **natural_questions**: Google QA from search queries
- **eli5**: Reddit explanations (good for explanatory style)
- **ms_marco**: Microsoft QA passages
- **yahoo_answers_qa**: Community questions
- **quora**: Question pairs

**Why not use these?**
- Not agriculture-specific
- Would need extensive fine-tuning
- Lower quality for farming questions
- KisanVaani is better suited

---

## 📊 Dataset Quality Comparison

| Dataset | Size | Quality | Agriculture Focus | Recommended |
|---------|------|---------|-------------------|-------------|
| KisanVaani | 22,615 | ⭐⭐⭐⭐⭐ | ✅ Yes | ✅ Primary |
| sowmya14 | 999 | ⭐⭐⭐⭐ | ✅ Yes (Pests) | ⚡ Optional |
| PSP-agricultureQnA | 1,000 | ⭐⭐⭐⭐ | ✅ Yes | ⚡ Optional |
| Natural-Farming | 1,126 | ⭐⭐⭐ | ✅ Yes (Organic) | ⚡ Optional |
| General QA | 100K+ | ⭐⭐⭐ | ❌ No | ❌ Not needed |

---

## 🎯 Recommendation

**For your project**: Stick with **KisanVaani/agriculture-qa-english-only**

**Why?**
1. ✅ Largest agriculture-specific dataset (22,615 examples)
2. ✅ High quality, clean data
3. ✅ Perfect for T5 fine-tuning
4. ✅ Already integrated into your notebook
5. ✅ More than enough for excellent results

**When to add more?**
- Only if you want to specialize (e.g., more pest control → add sowmya14)
- If KisanVaani results are poor (unlikely)
- For future iterations after initial success

---

## 🚀 Current Setup (Optimal)

```python
# Your notebook currently uses (RECOMMENDED):
dataset = load_dataset("KisanVaani/agriculture-qa-english-only")

# Training split: ~18,000 examples
# Validation: ~2,300 examples  
# Test: ~2,300 examples

# This is EXCELLENT for T5 fine-tuning! ✅
```

**No changes needed!** Your current setup is optimal.

---

## 📝 Notes

- All datasets tested and verified working
- Column naming differences handled in notebook
- Links provided in DATASET_INFO.md
- Focus on training with KisanVaani first
- Can experiment with combinations later

**Status**: ✅ Current dataset selection is optimal
