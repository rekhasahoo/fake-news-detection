import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# Load dataset
dataset_path = 'dataset.csv'  # Update this path if needed
data = pd.read_csv(dataset_path)
data['news'] = data['news'].fillna("")
X = data['news']
y = data['label']

# Define multiple models
models = {
    'Logistic Regression': Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', LogisticRegression(max_iter=1000))
    ]),
    'Random Forest': Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', RandomForestClassifier(n_estimators=100))
    ]),
    'Naive Bayes': Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB())
    ]),
    'SVM': Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', LinearSVC())
    ])
}

# Train all models and store results
results = []

for name, model in models.items():
    model.fit(X, y)
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    results.append({
        'Model': name,
        'Accuracy': round(acc, 4)
    })

# Convert to DataFrame and print
results_df = pd.DataFrame(results)
print("\n🔍 Model Accuracy Comparison on Training Data:\n")
print(results_df.to_string(index=False))
