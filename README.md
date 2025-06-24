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
- 2D CNN
- MLP

### Embeddings
- Pre-trained sentence-transformers
- **Fine-tuned embedding classifier** (best model)

## Results

These are filler values (Need to update)
| Model | Accuracy | F1-Score |
|-------|----------|----------|
| Logistic Regression | ~85% | 0.84 |
| XGBoost | ~88% | 0.88 |
| 1D CNN | ~93% | 0.93 |
| MLP | ~92% | 0.92 |
| **Fine-tuned Embeddings** | **96%** | **0.96** |

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
