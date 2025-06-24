import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from transformers import AutoModel, AutoTokenizer
import os
import platform
import psutil

device = 'cuda' if torch.cuda.is_available() else 'cpu'

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

class EmbeddingClassifier(nn.Module):
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        super(EmbeddingClassifier, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        self.classifier = nn.Sequential(
            nn.Linear(384, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 2)
        )
    
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )
    
    def forward(self, input_texts):
        encoded = self.tokenizer(
            input_texts,
            padding=True,
            truncation=True,
            return_tensors='pt'
        ).to(next(self.parameters()).device)
        
        model_output = self.encoder(**encoded)
        sentence_embeddings = self.mean_pooling(model_output, encoded['attention_mask'])
        logits = self.classifier(sentence_embeddings)
        return logits

@st.cache_resource
def load_model():
    """Load the pre-trained model and updated/finetuned weights"""
    try:
        model = EmbeddingClassifier()
        model_path = "bestModel.pth"
        
        state_dict = torch.load(model_path, map_location=device)  # Load state dict
        model.load_state_dict(state_dict)
        model.to(device) 
        model.eval()
        
        return model, device
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def get_system_info():
    """Get system information including CPU, RAM, and GPU details"""
    try:
        # info about cpu, ram and and other basic system details
        system_info = {
            'OS': platform.system(),
            'OS Version': platform.version(),
            'Architecture': platform.architecture()[0],
            'Processor': platform.processor(),
            'CPU Cores': psutil.cpu_count(logical=False),
            'CPU Threads': psutil.cpu_count(logical=True),
            'RAM Total': f"{psutil.virtual_memory().total / (1024**3):.1f} GB",
            'RAM Available': f"{psutil.virtual_memory().available / (1024**3):.1f} GB",
            'RAM Usage': f"{psutil.virtual_memory().percent}%"
        }
        
        # GPU information
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            current_gpu = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(current_gpu)
            gpu_memory = torch.cuda.get_device_properties(current_gpu).total_memory / (1024**3)
            
            system_info.update({
                'GPU Available': 'Yes',
                'GPU Count': gpu_count,
                'Current GPU': gpu_name,
                'GPU Memory': f"{gpu_memory:.1f} GB"
            })
        else:
            system_info.update({
                'GPU Available': 'No',
                'GPU Count': 0,
                'Current GPU': 'None',
                'GPU Memory': 'N/A'
            })
            
        return system_info
    except Exception as e:
        return {'Error': str(e)}

def predict_news(model, title: str, text: str):
    """Given a title/text, predict whether it is fake/real"""
    try:
        # user can pass either title, text, or both but at least one must be filled
        if pd.isna(title) or title.strip() == "":
            input_text = text
        elif pd.isna(text) or text.strip() == "":
            input_text = title
        else:
            input_text = f"{title} {text}"
        
        with torch.no_grad():
            logits = model([input_text])
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(logits, dim=1)
        
        probs = probabilities.cpu().numpy()[0]
        prediction = predicted_class.cpu().numpy()[0]
        return prediction, probs
    
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None, None

def main():
    st.title("Fake News Detection System")
    st.markdown("---")
    
    st.markdown("""
    This application uses a fine-tuned embedding classifier to detect whether a news article is **Real** or **Fake**.
    The model is based on a sentence transformer (all-MiniLM-L6-v2) with an appened  MLP classifier at the end. Entire model was fine-tuned on with a very small learning rate for 5 epochs. 
    
    **Instructions:**
    - Enter either a title, text content, or both
    - At least one field must be filled
    - Click 'Analyze News' to get the prediction
    """)
    
    model, device = load_model()
    
    if model is None:
        st.error("Failed to load the model. problem with model file.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 News Title")
        title = st.text_area(
            "Enter the news title:",
            height=100,
            placeholder="Enter the headline or title of the news article..."
        )
    
    with col2:
        st.subheader("📄 News Content")
        text = st.text_area(
            "Enter the news content:",
            height=100,
            placeholder="Enter the main content or body of the news article..."
        )
    
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
                result_col1, result_col2, result_col3 = st.columns([1, 2, 1])
                
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
                    
                    # additional optics (green to red = real to fake)
                    confidence = max(real_prob, fake_prob)
                    if confidence >= 0.8:
                        confidence_level = "Very High"
                        confidence_color = "green"
                    elif confidence >= 0.65:
                        confidence_level = "High"
                        confidence_color = "orange"
                    elif confidence >= 0.55:
                        confidence_level = "Medium"
                        confidence_color = "yellow"
                    else:
                        confidence_level = "Low"
                        confidence_color = "red"
                    
                    st.markdown(f"**Overall Confidence:** :{confidence_color}[{confidence_level} ({confidence:.2%})]")
    
    st.markdown("---")
    st.markdown("### ℹ️ About the Model")
    
    with st.expander("Model Details"):
        st.markdown("""
        **Architecture:**
        - Base Model: sentence-transformers/all-MiniLM-L6-v2
        - Appended Classifier: Multi-layer Perceptron (MLP) (2x hidden layers with 128 neurons each)
        - Layers: 384 → 128 → 128 → 2 
        - Activation: LeakyReLU
        
        **Training:**
        - Fine-tuned on fake news dataset (entire model not just appended new classification layers).
        - Achieved ~96% accuracy on test set
        - Uses mixed text content (title + article body)
        
        **Output:**
        - 0: Real News
        - 1: Fake News  
        """)
      # info about system is being added to the sidebar which will run when the site is opened first
    with st.sidebar:
        st.header("🤖 Model Information")
        
        if model is not None:
            st.success("✅ Model Loaded Successfully")
            
            if device == 'cuda':
                st.info(f"🚀 Running on GPU: {torch.cuda.get_device_name(0)}")
            else:
                st.info("💻 Running on CPU")
            
            total_params = sum(p.numel() for p in model.parameters())            
            st.metric("Total Parameters", f"{total_params:,}")
        else:
            st.error("Make sure your model is loaded")
        
        st.markdown("---")
        st.header("💻 System Information")
        
        sys_info = get_system_info()
        
        if 'Error' not in sys_info:
            st.markdown("**Hardware:**")
            st.text(f"OS: {sys_info['OS']}")
            st.text(f"Architecture: {sys_info['Architecture']}")
            st.text(f"CPU Cores: {sys_info['CPU Cores']}")
            st.text(f"CPU Threads: {sys_info['CPU Threads']}")
            
            st.markdown("**Memory:**")
            st.text(f"Total RAM: {sys_info['RAM Total']}")
            st.text(f"Available RAM: {sys_info['RAM Available']}")
            st.text(f"RAM Usage: {sys_info['RAM Usage']}")
            
            st.markdown("**GPU:**")
            if sys_info['GPU Available'] == 'Yes':
                st.text(f"GPU: {sys_info['Current GPU']}")
                st.text(f"GPU Memory: {sys_info['GPU Memory']}")
                st.text(f"GPU Count: {sys_info['GPU Count']}")
            else:
                st.text("GPU: Not Available")
        else:
            st.error(f"Error getting system info: {sys_info['Error']}")

if __name__ == "__main__":
    main()
