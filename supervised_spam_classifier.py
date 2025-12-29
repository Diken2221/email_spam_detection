import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# load dataset
data = pd.read_csv("emails.csv", encoding="latin-1")[["label", "message"]]

# convert labels to numbers
data["label"] = data["label"].map({"ham": 0, "spam": 1})

X = data["message"]
y = data["label"]

# split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# text vectorization
vectorizer = TfidfVectorizer(stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# test model
y_pred = model.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# confusion matrix
disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.show()

# user input prediction
email = input("\nEnter email text to check: ")
email_vec = vectorizer.transform([email])
result = model.predict(email_vec)

if result[0] == 1:
    print("Result: Spam")
else:
    print("Result: Not Spam")
