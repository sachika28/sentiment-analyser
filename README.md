# Sentiment Analyser 

An NLP project that detects whether a movie review is Positive or Negative.

## Model Accuracy
✅ 86.40% accuracy using Logistic Regression + TF-IDF

## What it Does
- Loads and cleans 10,000 real IMDB movie reviews
- Removes HTML tags, special characters and stopwords
- Converts text to numbers using TF-IDF Vectorizer
- Trains a Logistic Regression model to classify sentiment
- Predicts sentiment on brand new reviews

## Tools Used
- Python
- Pandas
- NLTK
- Scikit-learn
- Matplotlib

## How to Run
1. Clone this repository
2. Download IMDB Dataset from Kaggle and rename to `imdb.csv`
3. Install requirements: `pip install pandas nltk scikit-learn matplotlib`
4. Run: `python sentiment.py`

## Sample Predictions
😊 "Brilliant acting and a gripping storyline!" → POSITIVE  
😞 "Boring and completely disappointing movie!" → NEGATIVE  

## Author
Sachika S
B.Sc. AI & Data Science Student 
