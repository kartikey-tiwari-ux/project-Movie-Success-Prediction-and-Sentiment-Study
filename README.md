🎬 Movie Success Prediction & Sentiment Study

This project predicts movie review sentiment (positive/negative) using IMDB dataset reviews and builds a simple Streamlit web app to test the model interactively.

📌 Project Overview

Objective:

Predict the sentiment of movie reviews (positive/negative).

Study genre-wise sentiment trends (future enhancement).

Provide an interactive web app for real-time predictions.

Tech Stack:

Python 🐍

Scikit-learn ⚡

NLTK (VADER, preprocessing)

Pandas, NumPy 📊

Streamlit 🌐

📂 Repository Structure
📁 movie-sentiment-app
 ├── app.py                 # Streamlit app script
 ├── sentiment_model.pkl    # Trained ML model (Logistic Regression/other)
 ├── tfidf_vectorizer.pkl   # TF-IDF vectorizer
 ├── requirements.txt       # Dependencies
 ├── README.md              # Project documentation

⚙️ Installation & Setup
1️⃣ Clone the repository
git clone https://github.com/kartikey-tiwari-ux/project-Movie-Success-Prediction-and-Sentiment-Study.git
cd movie-sentiment-app

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run locally with Streamlit
streamlit run app.py


Open in browser: 👉 http://localhost:8501

🚀 Deployment (Streamlit Cloud)

Push this repo to GitHub.

Go to Streamlit Cloud
.

Create new app → select repo → choose app.py.

Add requirements.txt.

Deploy 🎉.

After deployment, you’ll get a public link like:

[https://your-username-movie-sentiment-app.streamlit.app](https://project-movie-success-prediction-and-sentiment-study-wltfqlko8.streamlit.app/?embed_options=dark_theme,show_colored_line,show_padding,show_toolbar)

🧠 Model Details

Preprocessing: Tokenization, TF-IDF vectorization.

Model: Logistic Regression (can be swapped with Naive Bayes/SVM).

Output: Sentiment → Positive / Negative.

📊 Future Enhancements

📈 Genre-wise sentiment analysis.

🎯 Predicting box office success using regression.

📱 Mobile-optimized design.
