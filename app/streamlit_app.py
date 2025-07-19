import streamlit as st


st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide", initial_sidebar_state="expanded")

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from llm import return_prompt, run_groq_summary

import os
import platform
import psutil

# '''
# adding this solves the issue given below but no idea why this is happening (works locally though)
# 2025-06-24 22:11:51.127 Examining the path of torch.classes raised: Tried to instantiate class '__path__._path', but it does not exist! Ensure that it is registered via torch::class_

# https://discuss.streamlit.io/t/message-error-about-torch/90886/5
# '''
torch.classes.__path__ = [] 

model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)

@st.cache_resource
def load_model():
    """Load the pre-trained model and updated/finetuned weights"""
    try:
        # setting up embedding model with LoRA; model will run on CPU
        base_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,
            id2label={0: "Real", 1: "Fake"},
            label2id={"Real": 0, "Fake": 1}
        )
        model = PeftModel.from_pretrained(base_model, "app/LoRA")
        model.eval()
        
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def perform_summarization(title: str, text: str, combined_text: str) -> str:
    '''
    If the text is larger than 512 tokens, we will perform summrization using an LLM.

    Args:
        title (str): The title of the news article.
        text (str): The original text of the news article.
        combined_text (str): The combined text of title and article body.
    Returns:
        str: The combined text after summarization if needed, otherwise the original combined text.
    '''
    num_combined_tokens = len(tokenizer.encode(combined_text, return_tensors='pt')[0])
    if num_combined_tokens > 512:
        print(f"Combined text is {num_combined_tokens} tokens, which exceeds the 512 token limit. Summarizing...")
        title_token_count = len(tokenizer.encode(title, return_tensors='pt')[0])
        text_token_count = len(tokenizer.encode(text, return_tensors='pt')[0])
        prompt = return_prompt(
            title=title,
            text=text,
            title_token_count=title_token_count,
            text_token_count=text_token_count,
            available_tokens_for_summarization = 512 - title_token_count
        )

        summarized_text = run_groq_summary(prompt, combined_text, max_tokens=512 - title_token_count)
        print("summarized_text: \n", summarized_text)
        combined_text = f"{title} {summarized_text}"

    return combined_text


def predict_news(model, title: str, text: str):
    """Given a title/text, predict whether it is fake/real
    Args:
        model: The pre-trained model loaded with LoRA weights.
        title (str): The title of the news article.
        text (str): The content of the news article.    
    Returns:
        tuple: A tuple containing the predicted class (0 for Real, 1 for Fake) and the probabilities for each class.
    """
    try:
        input_text = f"{title} {text}"
        print(title)
        # check if input text is too long or not, and perform summarization if needed
        input_text = perform_summarization(title, text, input_text)

        # extracting input_ids and attention_mask from the tokenizer and making predictions
        with torch.no_grad():
            encoding = tokenizer(input_text, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
            input_ids = encoding["input_ids"].squeeze(1)
            attention_mask = encoding["attention_mask"].squeeze(1)
            logits = model(input_ids, attention_mask=attention_mask).logits
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(logits, dim=1)
        
        # getting predicted probabilities
        probs = probabilities.numpy()[0]
        prediction = predicted_class.numpy()[0]
        return prediction, probs
    
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None, None

def main():
    st.title("📰 Fake News Detection System")
    st.markdown("#### ℹ️ About the Model")
    st.info(
        """
        This app uses a fine-tuned embedding model with Low-Rank Adaptation (LoRA) to classify news articles as **Real** or **Fake**.

        **Model Details**
        - Base Model: sentence-transformers/all-MiniLM-L6-v2
        - Finetuning: Low-Rank Adaptation (LoRA)
        - Training:
            - Fine-tuned on a fake news dataset
            - ~97% accuracy on test set, 99% on additional test data
            - Uses mixed text (title + article body) and llama-3.3-70b for summarizing huge articles
        """
    )
    
    model = load_model()
    
    if model is None:
        st.error("Failed to load the model. problem with model file.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 News Title")
        title = st.text_area("Enter the news title:", height=100, placeholder="Enter title here..")
    
    with col2:
        st.subheader("📄 News Content")
        text = st.text_area("Enter the news content:", height=100, placeholder="Enter news article here..")
    
    st.markdown("---")
    
    if st.button("🔍 Analyze News", type="primary", use_container_width=True):
        # one of title/text must be filled
        title_empty = pd.isna(title) or title.strip() == ""
        text_empty = pd.isna(text) or text.strip() == ""
        
        if title_empty and text_empty:
            st.error("⚠️ Please enter either a title, content, or both. At least one field must be filled.")
        else:
            # spinner object while the script predicts the model as some CPU might take time (just for user friendliness)
            with st.spinner("Analyzing the news article..."):
                prediction, probabilities = predict_news(model, title, text)
            
            if prediction is not None and probabilities is not None:
                st.markdown("---")
                st.subheader("📊 Analysis Results")
                _, result_col2, _ = st.columns([1, 2, 1])
                
                with result_col2:
                    if prediction == 0:
                        st.success("✅ **REAL NEWS**")
                        st.balloons()
                    else:
                        st.error("❌ **FAKE NEWS**")
                    
                    st.markdown("### Confidence Scores:")
                    real_prob = float(probabilities[0])
                    fake_prob = float(probabilities[1])
                    
                    st.metric("Real News Probability", f"{real_prob:.2%}")
                    st.progress(real_prob)
                    
                    st.metric("Fake News Probability", f"{fake_prob:.2%}")
                    st.progress(fake_prob)
                    
                    # we decided to include some additional colored  optics  to make it look more appealing (green to red = real to fake)
                    confidence = max(real_prob, fake_prob)
                    if confidence >= 0.9:
                        confidence_level = "Very High"
                        confidence_color = "green"
                    elif confidence >= 0.75:
                        confidence_level = "High"
                        confidence_color = "orange"
                    elif confidence > 0.5:
                        confidence_level = "Medium"
                        confidence_color = "yellow"
                    else:
                        confidence_level = "Low"
                        confidence_color = "red"
                    
                    st.markdown(f"**Overall Confidence:** :{confidence_color}[{confidence_level} ({confidence:.2%})]")
    
    st.markdown("---")
    
if __name__ == "__main__":
    main()
