import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Step 1: Ask for file input
file_name = input("Enter the CSV file name (with extension): ")

try:
    df = pd.read_csv(file_name)
except FileNotFoundError:
    print("❌ File not found. Please check the file name and try again.")
    exit()

# Step 2: Auto-detect label column
label_column = None
for col in df.columns:
    if col.lower() in ['class', 'fraud', 'label', 'is_fraud']:  # now includes is_fraud
        label_column = col
        break

if not label_column:
    print("❌ Dataset must contain a label column like 'Class', 'fraud', 'label', or 'is_fraud'.")
    print("Available columns:", list(df.columns))
    exit()

print(f"✅ Detected label column: '{label_column}'")

# Step 3: Prepare features and labels
X = df.drop(label_column, axis=1)
y = df[label_column]

# Drop non-numeric columns for simplicity
X = X.select_dtypes(include=['int64', 'float64'])

# Step 4: Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)

# Step 6: Build and train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 7: Make predictions
y_pred = model.predict(X_test)

# Step 8: Evaluation
print("\n--- 📊 Model Evaluation ---")
print("✅ Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"]))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Step 9: Predict for all data and label
df['Predicted_Class'] = model.predict(X_scaled)
df['Prediction_Label'] = df['Predicted_Class'].map({0: 'Legitimate', 1: 'Fraud'})

# Step 10: Show sample output
print("\n--- 🔍 Sample Predictions ---")
print(df[['Predicted_Class', 'Prediction_Label']].head(60))

# Step 11: Save to new file
output_file = "fraud_detection_results.csv"
df.to_csv(output_file, index=False)
print(f"\n✅ Predictions saved to: {output_file}")
