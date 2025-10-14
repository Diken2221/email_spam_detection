# Email Spam Classifier

This project demonstrates a simple email classification system using Python and scikit-learn. It allows you to classify emails as spam or ham using either supervised or unsupervised learning approaches.

Features:
- Supervised Learning: Uses Multinomial Naive Bayes for classification.
- Unsupervised Learning: Uses KMeans clustering to group emails.
- Converts email text to numerical features using TfidfVectorizer.

Requirements:
- Python 3.x
- pandas
- scikit-learn

Install required packages using:
pip install pandas scikit-learn

How to Use:
1. Place your email dataset in a CSV file named emails.csv with the following columns:
   - label (values: ham or spam)
   - message (email text)
2. Run the script:
python email_classifier.py
3. Follow the prompts:
   - Enter 1 for supervised learning or any other key for unsupervised learning.
   - Input the email text you want to check.
4. The model will output the prediction:
   - 0 for ham (not spam)
   - 1 for spam (for supervised learning)
   - For unsupervised learning, the output is the cluster index (0 or 1).

How It Works:
1. Data Loading: Loads email data from emails.csv.
2. Preprocessing: Converts email messages to numeric vectors using TfidfVectorizer.
3. Model Training:
   - Supervised: Trains MultinomialNB on labeled data.
   - Unsupervised: Uses KMeans clustering on email vectors.
4. Prediction: Checks a new email and outputs the result.

Usage Example:
Enter 1 for supervised else any key for unsupervised = 1
Enter your Email which you want to check = "Win cash prizes now!"
[1]

Notes:
- Supervised learning requires labeled data (ham or spam).
- Unsupervised learning clusters emails without labels, so output may need interpretation.
