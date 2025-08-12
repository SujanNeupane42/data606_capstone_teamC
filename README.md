# DATA 606 Capstone Project - Team C
## Fake News Detection

Binary classification of news articles as real or fake using various ML/DL approaches.

## Data
- **WELFake Dataset**: 72,134 articles
- **Kaggle Fake News Dataset**: Combined true/fake news
- **News Sentiment Dataset**: Additional features
- **Final Split**: 49,043 train / 8,655 val / 14,425 test
- **Preprocessing**: Text cleaning, LLM summarization (Llama 3.2 3B), title+content combination
- **Additional independent Final Testing Data with 45k news instances for final robust testing**
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


## Results on final testing data

| Model Name                               | Precision | Recall | F1 Score | Accuracy | AUC Score | FPs   | FNs   |
|------------------------------------------|-----------|--------|----------|----------|-----------|-------|-------|
| Logistic Regression                      | 0.905     | 0.905  | 0.90     | 0.90     | 0.9026    | 3082  | 1359  |
| XGBoost                                  | 0.965     | 0.965  | 0.96     | 0.96     | 0.9941    | 1202  | 528   |
| 1-d CNN                                  | 0.8850    | 0.88   | 0.88     | 0.88     | 0.9554    | 3903  | 1522  |
| 2-d CNN                                  | 0.87      | 0.87   | 0.87     | 0.87     | 0.9443    | 3465  | 2437  |
| MLP                                      | 0.925     | 0.93   | 0.93     | 0.93     | 0.9819    | 2416  | 886   |
| Full-Finetuned Embedding Classifier      | 0.99      | 0.99   | 0.99     | 0.99     | 0.9996    | 35    | 488   |
| **LoRA-Finetuned Embedding Classifier**  | **0.99**  | **0.99** | **0.99** | **0.99** | **0.9997**| **59**| **432**|


Table for macro average precision, recall, f1 score, accuracy, AUC score, and false positives and negatives count for each classification model prediction on the final independent test set. This test dataset has 21417 negative or real news samples and 23481 fake news or positive samples.
<br><br>

![Confusion Matrix for LoRA-Embedding Classifier](image.png)
<br>
Confusion matrix for the LoRA-Embedding classifier model based on its predictions on the final test set, representing TNs, TPs, FPs, and FNs.

|              | Precision | Recall | F1 Score | Support |
|--------------|-----------|--------|----------|---------|
| 0 (real)     | 0.98      | 1.00   | 0.99     | 21417   |
| 1 (fake)     | 1.00      | 0.98   | 0.99     | 23481   |
| **Accuracy** |           |        | **0.99** | 44898   |
| Macro avg    | 0.99      | 0.99   | 0.99     | 44898   |
| Weighted avg | 0.99      | 0.99   | 0.99     | 44898   |

<br>
Classification report for LoRA-Embedding classifier on the final test set, with 0 indicating real news and 1 indicating fake news. Here, Recall and Precision for classes 0 and 1 are 1 due to scikit-learn's classification report function rounding off values to the next nearest value beyond 2nd precision. We keep the results exactly as we got with scikit-learn’s function.


## Streamlit App

The fake news detection app is deployed and accessible at: [https://fakenewsweb.streamlit.app/](https://fakenewsweb.streamlit.app/)

To run locally:
Create and activate a virtual env fist, and install dependencies from requirements.txt, and execute the following:
```bash
cd app
streamlit run streamlit_app.py
```

## Team
- Sujan Neupane (bv23415)
- Danny McKirgan
- Mindy Shang
