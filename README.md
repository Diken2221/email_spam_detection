# 📧 Email Spam Detection

This project implements an **Email Spam Detection System** using both **Supervised** and **Unsupervised Machine Learning** techniques.  
The goal is to classify emails as **Spam** or **Ham (Not Spam)** and to explore how different learning paradigms perform on the same problem.

---

## 🚀 Features

- Text preprocessing (cleaning, tokenization, stopword removal)
- Feature extraction using **TF-IDF**
- **Supervised Learning Models**
  - Naive Bayes
  - Logistic Regression
  - Support Vector Machine (SVM)
- **Unsupervised Learning Models**
  - K-Means Clustering
  - Hierarchical Clustering (optional)
- Model evaluation and comparison
- Visualization of results

---

## 🧠 Machine Learning Approaches

### 1️⃣ Supervised Learning
Supervised models are trained using **labeled email data** (spam or ham).

**Algorithms Used:**
- Multinomial Naive Bayes
- Logistic Regression
- Support Vector Machine (SVM)

**Advantages:**
- High accuracy with labeled data
- Clear evaluation metrics

---

### 2️⃣ Unsupervised Learning
Unsupervised models work **without labeled data** and group emails based on similarity.

**Algorithms Used:**
- K-Means Clustering

**Approach:**
- Emails are clustered into groups
- Clusters are analyzed and mapped to spam/ham labels

**Advantages:**
- Useful when labeled data is unavailable
- Helps discover hidden patterns

---


