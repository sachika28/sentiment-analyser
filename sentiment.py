import pandas as pd
import nltk
import re
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

nltk.download('stopwords')
from nltk.corpus import stopwords

# Load dataset
print("Loading dataset...")
df = pd.read_csv("imdb.csv")
print(f"✅ Loaded {len(df)} reviews!")

# Clean text function
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)        # remove HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', text) # remove special characters
    text = text.lower()                       # lowercase
    words = text.split()
    stops = set(stopwords.words('english'))
    words = [w for w in words if w not in stops]
    return ' '.join(words)

print("Cleaning text...")
df['clean_review'] = df['review'].apply(clean_text)
print("✅ Text cleaned!")

# Use 10,000 reviews for speed
df = df.sample(10000, random_state=42)

# Convert to numbers
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['clean_review'])
y = df['sentiment']

# Split and train
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print("Training model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Accuracy
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"\n✅ Model Accuracy: {accuracy * 100:.2f}%")
print("\nDetailed Report:")
print(classification_report(y_test, predictions))

# Test with custom reviews
my_reviews = [
    "This movie was absolutely amazing and wonderful!",
    "I hated this film, it was a terrible waste of time!",
    "Brilliant acting and a gripping storyline!",
    "Boring and completely disappointing movie!",
    "One of the best films I have ever seen!",
]

print("\n🎬 Testing with new reviews:")
for review in my_reviews:
    cleaned = clean_text(review)
    result = model.predict(vectorizer.transform([cleaned]))[0]
    emoji = "😊" if result == "positive" else "😞"
    print(f"{emoji} '{review}'")
    print(f"   → {result.upper()}\n")

# Chart
plt.figure(figsize=(6, 4))
df['sentiment'].value_counts().plot(kind='bar', color=['green', 'red'])
plt.title("Sentiment Distribution (10,000 Reviews)")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("sentiment_chart.png")
plt.show()

print("🎉 Sentiment Analyser Complete!")