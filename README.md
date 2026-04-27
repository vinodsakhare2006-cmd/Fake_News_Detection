# 📰 Fake News Detection App (Streamlit + Machine Learning)

A fast and lightweight **Fake News Detection Web App** built using **Natural Language Processing (NLP)** and **Machine Learning**, deployed with **Streamlit**.

---

## 🚀 Live Demo

👉 Add your Streamlit link here after deployment.

---

## 📌 Project Overview

Fake news is a major problem in today's digital world. This project uses **Machine Learning + NLP** to classify news as:

- ✅ **Real News**
- ❌ **Fake News**

The model is trained on labeled news data and deployed as an interactive web application.

---

## ⚡ Features

- 🧠 ML-based classification
- ⚡ Fast Streamlit UI
- 📊 Confidence score display
- 📝 Text input for custom news checking
- 🎯 Lightweight and deployment-friendly
- 🔄 Easy retraining with new datasets
- 📁 Dataset included

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Web App | Streamlit |
| ML Model | Passive Aggressive Classifier |
| NLP | TF-IDF Vectorizer |
| Libraries | Scikit-learn, Pandas, NumPy |

---

## 📂 Project Structure

```text
Fake_News_Detection/
│
├── app.py
├── train_model.py
├── model.pkl
├── vectorizer.pkl
├── metrics.json
├── requirements.txt
├── README.md
├── notebook/
│   └── fake_news_detection.ipynb
└── dataset/
    ├── news.csv
    ├── True.csv
    └── Fake.csv
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/Fake_News_Detection.git
cd Fake_News_Detection
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run Application

```bash
streamlit run app.py
```

---

## 🌐 Deployment on Streamlit Cloud

1. Push this project to GitHub
2. Go to https://streamlit.io/cloud
3. Click **New App**
4. Select your repository
5. Set main file path as:

```text
app.py
```

6. Click **Deploy**

---

## 📊 Model Details

- **Algorithm:** Passive Aggressive Classifier
- **Vectorizer:** TF-IDF
- **Output Labels:**
  - `1` → Real News
  - `0` → Fake News

---

## 🧪 Sample Test Cases

### ❌ Fake News Example

```text
NASA plans to build a luxury hotel on the Sun by 2035.
```

### ✅ Real News Example

```text
Scientists develop AI tool for early disease detection.
```

---

## 🔄 Retrain Model

To retrain the model:

```bash
python train_model.py
```

This will regenerate:

```text
model.pkl
vectorizer.pkl
metrics.json
```

---

## 📌 Dataset Note

This project includes a small demo dataset so the app works immediately.

For better accuracy, replace the files inside the `dataset/` folder with the Kaggle/ISOT Fake and Real News Dataset:

- `True.csv`
- `Fake.csv`

Then run:

```bash
python train_model.py
```

---

## ⚠️ Disclaimer

This project is for **educational purposes only**.  
Always verify important news from trusted and official sources.

---

## 👨‍💻 Author

**Kailash Dake**

---

## ⭐ Support

If you like this project, please give it a ⭐ on GitHub.
