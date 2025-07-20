# DATA 606 Capstone Project - Team C
## Fake News Detection

Binary classification of news articles as real or fake using various ML/DL approaches.

## Data
- **WELFake Dataset**: 72,134 articles
- **Kaggle Fake News Dataset**: Combined true/fake news
- **News Sentiment Dataset**: Additional features
- **Final Split**: 49,043 train / 8,655 val / 14,425 test
- **Preprocessing**: Text cleaning, LLM summarization (Llama 3.2 3B), title+content combination

## Experiments

### Traditional ML
- Logistic Regression 
- XGBoost

### Deep Learning  
- 1D CNN
- 2D CNN (reshaping the embedding to a matrix to mimic a grey-scale image)
- MLP
- Full fine-tuned embedding classifier
- LoRA based Embedding-Classifier finetuning


## Results

| Model | Accuracy |
|-------|----------|
| Logistic Regression | 90% |
| XGBoost | 96% |
| 1D CNN | 88% |
| 2D CNN | 87% |
| MLP | 93% |
| Fine-tuned Embeddings | 99% |
| Fine-tuned Embeddings with LoRA | **99%** |

**Best Model**: sentence-transformers/all-MiniLM-L6-v2 + MLP (384→128→128→2)  
**ROC AUC**: 99.48%

## Streamlit App

```bash
cd app
streamlit run streamlit_app.py
```

## Team
- Sujan Neupane (bv23415)
- Danny McKirgan
- Mindy Shang
