import pickle

# Load saved files
tfidf = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

# Example input
text = ["This movie was fantastic and amazing"]

# Apply SAME preprocessing (IMPORTANT ⚠️)
# You must reuse your cleaning + spacy logic here

vec = tfidf.transform(text)
prediction = model.predict(vec)

print(prediction)