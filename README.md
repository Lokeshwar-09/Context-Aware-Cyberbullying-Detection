# 🛡️ Context-Aware Cyberbullying Detection using BERT

A machine learning system that utilizes a fine-tuned **BERT (Bidirectional Encoder Representations from Transformers)** model to detect cyberbullying and harassment in online conversations. Unlike traditional classifiers, this system is **context-aware**, meaning it analyzes both the **previous comment** (context) and the **reply comment** together to make highly accurate classifications.

---

## 🚀 Features

- **Context-Aware Classification**: Processes conversation pairs (Context + Reply) to detect sarcasm, replies triggered by harassment, and context-dependent bullying.
- **CyberGuard AI Dashboard**: A modern, interactive split-pane Streamlit web interface with responsive visualization.
- **Dynamic Category Highlighting**: Color-coded status updates (Green for Safe, Orange for Harassment, Red for Hate Speech, Purple for Sexual Bullying).
- **Proportional Metrics**: Real-time confidence metrics with custom-colored probability meters.
- **One-Click Safety Actions**: Direct integration to file official complaints with the Cybercrime Portal if harmful content is detected.
- **CLI Mode**: Fast command-line interface for quick testing and local scripting.

---

## 📁 Repository Structure

```text
├── app.py               # Streamlit web dashboard interface
├── predict.py           # CLI prediction and inference script
├── preprocess.py        # Dataset preprocessing utilities
├── train_bert.py        # BERT model training and fine-tuning pipeline
├── requirements.txt     # Python dependency configuration
└── .gitignore           # Git ignore configurations (ignores models, datasets, etc.)
```

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Lokeshwar-09/Context-Aware-Cyberbullying-Detection.git
cd Context-Aware-Cyberbullying-Detection
```

### 2. Set Up Virtual Environment (Recommended)
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Setup Model Weights
Place your trained model files (weights, config, and `label_encoder.pkl`) inside the `models/` directory:
```text
models/
├── config.json
├── label_encoder.pkl
├── model.safetensors
├── special_tokens_map.json
├── tokenizer_config.json
└── vocab.txt
```
*(Note: These files are automatically ignored by Git as they are too large for repository hosting.)*

---

## 💻 How to Run

### Run the Web Dashboard
Start the interactive Streamlit application in your browser:
```bash
streamlit run app.py
```

### Run CLI Prediction Tool
For running rapid predictions directly in your terminal:
```bash
python predict.py
```

---

## 🧠 Model Specifications

The classifier processes the text in a sentence-pair format:
`[CLS] Previous Comment (Context) [SEP] Reply Comment [SEP]`

The model is trained to output one of the following labels:
- **Safe (Not Cyberbullying)** ✅
- **Harassment Bullying** ⚠️
- **Hate Speech Bullying** 🚨
- **Sexual Bullying** 🔞

---

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/Lokeshwar-09/Context-Aware-Cyberbullying-Detection/issues).
