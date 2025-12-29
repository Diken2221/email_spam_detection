import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# read dataset
data = pd.read_csv("emails.csv", encoding="latin-1")

# keep required columns
data = data[['v1', 'v2']]
data.columns = ['label', 'text']

# convert labels to numbers
data['label'] = data['label'].map({'spam': 1, 'ham': 0})

# split data
X_train, X_test, y_train, y_test = train_test_split(
    data['text'],
    data['label'],
    test_size=0.2,
    random_state=42
)

# convert text to numerical form
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# test accuracy
predictions = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy * 100)

# user input checking
while True:
    email = input("Enter email text (exit to stop): ")

    if email == "exit":
        break

    email_vec = vectorizer.transform([email])
    result = model.predict(email_vec)

    if result[0] == 1:
        print("Spam")
    else:
        print("Not Spam")
