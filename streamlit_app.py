import streamlit as st
import torch
import pickle
import warnings
import logging
import os
from huggingface_hub import hf_hub_download

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers.models.bert.modeling_bert import BertForSequenceClassification


# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Cyberbullying Detection",
    page_icon="🛡️",
    layout="centered"
)


# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Load Model — cached so it
# loads only once
# -----------------------------
@st.cache_resource
def load_model(model_dir="models"):
    # Check if local folder exists, else load from Hugging Face Hub
    if not os.path.exists(model_dir) or not os.path.isdir(model_dir):
        # Default fallback or user specified secret repo name
        if "HF_MODEL_REPO" in st.secrets:
            model_dir = st.secrets["HF_MODEL_REPO"]
        else:
            model_dir = "Lokeshwar-09/context-aware-cyberbullying-detection"

    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model     = BertForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    
    if "/" in model_dir:
        label_encoder_path = hf_hub_download(repo_id=model_dir, filename="label_encoder.pkl")
    else:
        label_encoder_path = os.path.join(model_dir, "label_encoder.pkl")
        
    with open(label_encoder_path, "rb") as f:
        encoder = pickle.load(f)
    return model, tokenizer, encoder



# -----------------------------
# Predict Function
# -----------------------------
def predict(previous_comment, reply_comment, model, tokenizer, encoder):
    inputs = tokenizer(
        previous_comment,
        reply_comment,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=128
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    logits     = outputs.logits
    probs      = torch.softmax(logits, dim=1)
    confidence = probs.max().item()
    pred_idx   = torch.argmax(probs, dim=1).item()
    label      = encoder.inverse_transform([pred_idx])[0]

    # All class probabilities
    all_probs = {
        encoder.classes_[i]: round(probs[0][i].item() * 100, 2)
        for i in range(len(encoder.classes_))
    }

    return label, round(confidence * 100, 2), all_probs


# -----------------------------
# Label colours and icons
# -----------------------------
LABEL_CONFIG = {
    "not_cyberbullying":    {"color": "#10B981", "icon": "✅", "text": "Safe (Not Cyberbullying)"},
    "harassment_bullying":  {"color": "#F59E0B", "icon": "⚠️", "text": "Harassment Bullying"},
    "hate_speech_bullying": {"color": "#EF4444", "icon": "🚨", "text": "Hate Speech Bullying"},
    "sexual_bullying":      {"color": "#8B5CF6", "icon": "🔞", "text": "Sexual Bullying"},
}


# -----------------------------
# Custom Premium CSS Styling
# -----------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    /* Apply custom font */
    .stApp, .stMarkdown, p, span, label, input, textarea, button {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }
    
    /* Header layout styling */
    .header-container {
        text-align: center;
        padding-top: 10px;
        padding-bottom: 20px;
    }
    
    .logo-container {
        font-size: 3.5rem;
        filter: drop-shadow(0px 8px 16px rgba(133, 45, 246, 0.25));
        margin-bottom: 10px;
        animation: float 3s ease-in-out infinite;
        display: inline-block;
    }
    
    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-8px); }
        100% { transform: translateY(0px); }
    }
    
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 5px 0 10px 0;
        background: linear-gradient(135deg, #FF4B4B 0%, #852DF6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.02em;
    }
    
    .subtitle {
        color: #7A869A;
        font-size: 1rem;
        font-weight: 500;
        margin-bottom: 25px;
    }
    
    /* Styling input text areas */
    .stTextArea label {
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        color: var(--text-color) !important;
    }
    
    /* Premium input shadow & border transitions */
    .stTextArea textarea {
        border-radius: 12px !important;
        border: 1px solid rgba(128, 128, 128, 0.2) !important;
        transition: all 0.3s cubic-bezier(0.165, 0.84, 0.44, 1) !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #852DF6 !important;
        box-shadow: 0 0 0 3px rgba(133, 45, 246, 0.15) !important;
    }
    
    /* Make buttons premium and glowing */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #FF4B4B 0%, #852DF6 100%) !important;
        border: none !important;
        color: white !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        padding: 14px 28px !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 15px rgba(133, 45, 246, 0.2) !important;
        transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
        width: 100% !important;
    }
    
    div.stButton > button:first-child:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(133, 45, 246, 0.35) !important;
    }
    
    div.stButton > button:first-child:active {
        transform: translateY(0px) !important;
    }
    
    /* Info/Warning adjustments */
    .stAlert {
        border-radius: 12px !important;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------
# Header Hero Section
# -----------------------------
st.markdown(
    """
    <div class="header-container">
        <span class="logo-container">🛡️</span>
        <h1 class="main-title">CyberGuard AI</h1>
        <p class="subtitle">Context-Aware Cyberbullying Detection powered by BERT</p>
    </div>
    """,
    unsafe_allow_html=True
)


# Load model
try:
    model, tokenizer, encoder = load_model("models")
except Exception as e:
    st.error(f"### ❌ Failed to load model: {e}")
    st.markdown("""
    Since the trained model files are very large (approx. 438 MB), they are ignored by Git and not uploaded to GitHub.
    
    To fix this and deploy your app on Streamlit Cloud, you need to upload your model to **Hugging Face**:
    
    1. **Create a Hugging Face Account** at [huggingface.co](https://huggingface.co) if you don't have one.
    2. **Create a new model repository** named `context-aware-cyberbullying-detection` (public).
    3. **Upload the contents of your local `models/` folder** to the repository. The files required are:
       - `model.safetensors`
       - `config.json`
       - `vocab.txt`
       - `tokenizer_config.json`
       - `special_tokens_map.json`
       - `label_encoder.pkl`
    
    Alternatively, you can run the helper script `upload_to_hf.py` in your local directory to upload the model automatically:
    ```bash
    python upload_to_hf.py
    ```
    
    *Note: If your Hugging Face username is different from `Lokeshwar-09`, you can define your repository in your Streamlit App settings under **Secrets** (e.g. `HF_MODEL_REPO = "your-username/context-aware-cyberbullying-detection"`).*
    """)
    st.stop()



# -----------------------------
# Session State
# -----------------------------
if "result" not in st.session_state:
    st.session_state.result = None


# -----------------------------
# Main Columns Layout
# -----------------------------
col1, col2 = st.columns([1.1, 0.9], gap="large")

with col1:
    st.markdown("### 💬 Input Details")
    
    previous_comment = st.text_area(
        "Previous Comment (Context)",
        placeholder="Enter the previous comment in the conversation...",
        height=110,
        key="prev"
    )
    
    reply_comment = st.text_area(
        "Reply Comment (To Analyze)",
        placeholder="Enter the reply comment to classify...",
        height=110,
        key="reply"
    )
    
    detect_clicked = st.button("🔍 Analyze Text", use_container_width=True, type="primary")

    if detect_clicked:
        if not previous_comment.strip():
            st.warning("Please enter the Previous Comment.")
        elif not reply_comment.strip():
            st.warning("Please enter the Reply Comment.")
        else:
            with st.spinner("Analyzing..."):
                label, confidence, all_probs = predict(
                    previous_comment.strip(),
                    reply_comment.strip(),
                    model,
                    tokenizer,
                    encoder
                )
                st.session_state.result = {
                    "label": label,
                    "confidence": confidence,
                    "all_probs": all_probs
                }


with col2:
    st.markdown("### 🎯 Analysis Results")
    
    if st.session_state.result is not None:
        res = st.session_state.result
        label = res["label"]
        confidence = res["confidence"]
        all_probs = res["all_probs"]
        
        cfg = LABEL_CONFIG.get(label, {"color": "#333", "icon": "❓", "text": label})
        
        # Main result card
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, {cfg['color']}12, {cfg['color']}22);
                border: 1px solid {cfg['color']}44;
                border-left: 8px solid {cfg['color']};
                border-radius: 16px;
                padding: 20px;
                margin-bottom: 20px;
                box-shadow: 0 4px 15px {cfg['color']}10;
            ">
                <div style="display: flex; align-items: center; gap: 12px;">
                    <span style="font-size: 2.2rem;">{cfg['icon']}</span>
                    <div>
                        <h4 style="margin: 0; color: #888888; text-transform: uppercase; font-size: 10px; letter-spacing: 0.1em; font-weight: 700;">Detection Result</h4>
                        <h2 style="color: {cfg['color']}; margin: 0; font-size: 1.5rem; font-weight: 700; line-height: 1.2;">
                            {cfg['text']}
                        </h2>
                    </div>
                </div>
                <hr style="border: 0; border-top: 1px solid {cfg['color']}1a; margin: 16px 0;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 14px; font-weight: 500; color: var(--text-color);">Analysis Confidence</span>
                    <span style="font-size: 1.25rem; font-weight: 700; color: {cfg['color']};">{confidence}%</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Classification details header
        st.markdown("##### 📊 Category Confidence Scores")
        
        # Print probabilities sorted descending
        for cls, prob in sorted(all_probs.items(), key=lambda x: -x[1]):
            cfg2 = LABEL_CONFIG.get(cls, {"color": "#888", "text": cls})
            st.markdown(
                f"""
                <div style="margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between; font-size: 13px; font-weight: 600; margin-bottom: 3px;">
                        <span style="color: var(--text-color);">{cfg2['text']}</span>
                        <span style="color: {cfg2['color']};">{prob}%</span>
                    </div>
                    <div style="background-color: rgba(128, 128, 128, 0.15); border-radius: 10px; height: 8px; width: 100%; overflow: hidden;">
                        <div style="background-color: {cfg2['color']}; width: {prob}%; height: 100%; border-radius: 10px;"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        # Report button / Warning if needed
        if label != "not_cyberbullying" and confidence > 40:
            st.markdown(
                """
                <div style="
                    background-color: rgba(239, 68, 68, 0.08);
                    border: 1px solid rgba(239, 68, 68, 0.2);
                    border-radius: 12px;
                    padding: 16px;
                    margin-top: 20px;
                    margin-bottom: 12px;
                ">
                    <span style="font-weight: 700; color: #EF4444; display: block; margin-bottom: 6px; font-size: 14px;">⚠️ Safety Advisory</span>
                    <span style="font-size: 13px; color: #888888; display: block; line-height: 1.4;">
                        This response has been flagged as harmful. Consider reporting it or blocking the sender.
                    </span>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.link_button(
                "🚨 File Official Report (Cybercrime India)",
                "https://cybercrime.gov.in",
                use_container_width=True
            )
            
    else:
        # Beautiful Empty State
        st.markdown(
            """
            <div style="
                border: 2px dashed rgba(128, 128, 128, 0.15);
                border-radius: 16px;
                padding: 50px 20px;
                text-align: center;
                color: #888888;
                margin-top: 10px;
                height: 100%;
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
            ">
                <span style="font-size: 3rem; display: block; margin-bottom: 15px; filter: grayscale(30%);">🛡️</span>
                <strong style="font-size: 16px; color: var(--text-color);">Awaiting Analysis</strong>
                <p style="font-size: 13px; color: #888888; margin-top: 8px; max-width: 250px; line-height: 1.4;">
                    Enter the conversation details on the left or select an example preset, then click <strong>Analyze Text</strong>.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )


# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: grey; font-size: 13px;">
        Context-Aware Cyberbullying Detection using BERT
    </div>
    """,
    unsafe_allow_html=True
)