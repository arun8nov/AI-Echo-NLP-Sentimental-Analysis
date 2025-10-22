# 🧠 AI Echo – NLP Sentimental Analysis

AI Echo is an **end-to-end Natural Language Processing (NLP)** project designed to analyze text sentiment using machine learning and deep learning models.  
It features an **interactive Streamlit dashboard** and an **analytical Dash-based interface** for real-time insights, visualizations, and model predictions.

---

## 🚀 Features

- **Sentiment Classification** – Detects Positive, Negative, and Neutral sentiments.
- **Interactive Dashboards**
  - `app.py`: Streamlit-based front-end for user interaction.
  - `dash.py` & `dash1.py`: Dash-based data exploration and visualization.
- **Multiple Models Supported**
  - Traditional ML models (Logistic Regression, SVM, Random Forest)
  - Deep learning models (LSTM, BERT, Transformers)
- **Data Preprocessing Pipeline**
  - Text cleaning, lemmatization, stopword removal, and emoji handling.
- **Visualization Tools**
  - WordCloud, Seaborn, and Plotly-based sentiment distribution charts.
- **Balanced Classification**
  - Incorporates **SMOTE** for handling class imbalance.

---

## 🧩 Project Structure

```
AI-Echo-NLP-Sentimental-Analysis/
│
├── app.py              # Streamlit app for user-facing sentiment predictions
├── dash.py             # Dash visualization dashboard (data analytics)
├── dash1.py            # Alternate Dash dashboard (exploratory analysis)
├── requirements.txt    # Dependencies
├── data/               # Dataset (CSV or text files)
├── models/             # Saved ML/DL models (.pkl or .h5)
├── assets/             # Images, icons, and dashboard-related assets
└── README.md           # Project documentation
```

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/arun8nov/AI-Echo-NLP-Sentimental-Analysis.git
cd AI-Echo-NLP-Sentimental-Analysis
```

### 2. Create a Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
source venv/bin/activate   # On Mac/Linux
venv\Scripts\activate      # On Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🧠 Run the Applications

### ▶️ Streamlit Interface
```bash
streamlit run app.py
```

### 📊 Dash Dashboard
```bash
python dash.py  class and objects
```

### 🔍 Alternate Dashboard
```bash
python dash1.py  class and objects
```

Then open your browser at the displayed `localhost` or `127.0.0.1` URL.

---

## 📦 Requirements

Your `requirements.txt` includes:
```
pandas
numpy
matplotlib
seaborn
plotly
nbformat
scikit-learn
nltk
contractions
wordcloud
streamlit
imbalanced-learn
transformers
torch
tensorflow
keras
langdetect
emoji
```

---

## 📈 Workflow Overview

1. **Data Collection**  
   Import dataset (CSV, text, or API-based).

2. **Text Preprocessing**  
   - Expand contractions  
   - Remove stopwords, punctuations, and emojis  
   - Tokenize and lemmatize  
   - Detect and normalize language  

3. **Feature Engineering**  
   - TF-IDF, Word2Vec, or BERT embeddings  
   - Handle class imbalance using SMOTE  

4. **Model Training & Evaluation**  
   - Train models (Logistic Regression, LSTM, Transformer)  
   - Evaluate using accuracy, F1-score, precision, recall  

5. **Deployment**  
   - Interactive Streamlit & Dash apps for real-time sentiment analysis  

---

## 📊 Visualizations

- WordClouds for positive, negative, and neutral sentiments  
- Sentiment distribution plots  
- Model comparison charts (accuracy, confusion matrix)  
- Interactive dashboards using Plotly and Seaborn  

---

## 📚 Technologies Used

| Category | Tools / Libraries |
|-----------|-------------------|
| Programming | Python |
| ML Frameworks | scikit-learn, TensorFlow, Keras, PyTorch |
| NLP Tools | NLTK, Transformers, LangDetect, Contractions |
| Visualization | Matplotlib, Seaborn, Plotly, WordCloud |
| Web App | Streamlit, Dash |
| Others | Pandas, NumPy, Imbalanced-learn |

---

## 🧾 Example Use Case

Enter a text such as:

> “The product quality is amazing and delivery was super fast!”

AI Echo will classify it as **Positive**, showing related word frequency and sentiment visualization.

---

## 💡 Future Enhancements

- Integrate real-time social media data (Twitter API, YouTube comments).  
- Add multi-language sentiment support.  
- Implement fine-tuned transformer models (e.g., RoBERTa, DistilBERT).  
- Include topic modeling and emotion detection.

---

## 👨‍💻 Author

**Arunprakash B**  
🔗 [GitHub](https://github.com/arun8nov)  
🔗 [LinkedIn](https://www.linkedin.com/in/arun8nov)  
🌐 [Portfolio](https://crystal-acai-529.notion.site/Hey-there-I-am-Arunprakash-B-223fe4a17f8a80faa5abee1f246a06f1?pvs=143)

---

## 🪪 License

This project is open-source and available under the **MIT License**.

---

