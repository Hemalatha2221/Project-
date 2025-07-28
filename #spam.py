import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import os

# Ask for file name
file_name = input("Enter the file name (e.g., spam.csv or spam.tsv): ")

# Check extension
ext = os.path.splitext(file_name)[1]

try:
    # Load CSV or TSV accordingly
    if ext == ".csv":
        df = pd.read_csv(file_name, encoding='latin-1', on_bad_lines='skip')
    elif ext == ".tsv":
        df = pd.read_csv(file_name, sep='\t', encoding='latin-1', on_bad_lines='skip')
    else:
        raise ValueError("Unsupported file format. Please provide .csv or .tsv file.")

    # Use only first two columns
    df = df.iloc[:, :2]
    df.columns = ['label', 'message']

    # Map labels to 0 and 1
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})

    # Drop missing values
    df.dropna(inplace=True)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Train Naive Bayes model
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)

    # Predict
    y_pred = model.predict(X_test_tfidf)

    # Print evaluation metrics
    print("\n✅ Evaluation Results:")
    print("Accuracy :", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall   :", recall_score(y_test, y_pred))
    print("F1 Score :", f1_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Create results DataFrame
    results = pd.DataFrame({
        'Message': X_test,
        'Actual': y_test.map({0: 'Ham', 1: 'Spam'}).values,
        'Predicted': pd.Series(y_pred).map({0: 'Ham', 1: 'Spam'}).values
    }).reset_index(drop=True)

    # Filter and display only spam predictions
    only_spam = results[results['Predicted'] == 'Spam']

    print("\n🚨 Only Spam Predictions from Test Set:\n")
    if only_spam.empty:
        print("No spam messages were predicted.")
    else:
        print(only_spam[['Message', 'Actual', 'Predicted']].head(20).to_string(index=False))

except FileNotFoundError:
    print("❌ File not found. Please check the path.")
except ValueError as ve:
    print("❌", ve)
except Exception as e:
    print("❌ An error occurred:", e)
