import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.cluster import KMeans

# Load data 
data = pd.read_csv('emails.csv',encoding='latin-1')[['label','message']]  
data['label'] = data['label'].map({'ham':0,'spam':1})

X = data['message']
y = data['label']
 
# text to number
vectorizer = TfidfVectorizer(stop_words='english',lowercase=True)
X_vectorized = vectorizer.fit_transform(X)

# FROM HERE CHOOSE SUPERVISED OR UNSUPERVISED LEARNING
input("enter 1 for superviced else any key for unsuperviced = ")

if input == 1:

    ## superviced

    # Train model
    model = MultinomialNB()
    model.fit(X_vectorized, y)

    predictions = model.predict(vectorizer.transform([input("Enter your Email which you want to check = ")]))
    print(predictions)

else:

    ## unsuperviced

    # Train model
    model = KMeans(n_clusters=2)
    model.fit(X_vectorized)
    predictions = model.predict(vectorizer.transform([input("Enter your Email which you want to check = ")]))
    print(predictions)
