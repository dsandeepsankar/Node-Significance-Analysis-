import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("node_significance_analysis_data.csv")  # Ensure this file is in the same directory

# Drop Node_ID as it's an identifier
df = df.drop(columns=["Node_ID"])

# Encode categorical features
label_encoders = {}
categorical_cols = ["Community_Assignment", "Node_Type", "Node_Significance"]
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Features and target
X = df.drop("Node_Significance", axis=1)
y = df["Node_Significance"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": SVC(),
    "Naive Bayes": GaussianNB()
}

# Train and evaluate
accuracies = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    accuracies[name] = acc

# Plot accuracy comparison
plt.figure(figsize=(10, 6))
plt.bar(accuracies.keys(), accuracies.values(), color='skyblue')
plt.ylabel('Accuracy')
plt.title('Model Comparison on Node Significance Prediction')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("model_accuracy_comparison.png")
plt.show()

# ---- Predict for new input ----
input_data = {
    "Node_Degree": 52,
    "Betweenness": 0.28093451,
    "Closeness": 0.09028977,
    "Eigenvector": 0.877373072,
    "Community_Assignment": "Community_A",
    "Node_Type": "Type_2",
    "Page_Rank": 0.974394808
}

# Create DataFrame for input
input_df = pd.DataFrame([input_data])

# Encode input categorical features
for col in ["Community_Assignment", "Node_Type"]:
    input_df[col] = label_encoders[col].transform(input_df[col])

# Train best model again (Gradient Boosting)
best_model = GradientBoostingClassifier()
best_model.fit(X_train, y_train)

# Predict significance
predicted_encoded = best_model.predict(input_df)[0]
predicted_significance = label_encoders["Node_Significance"].inverse_transform([predicted_encoded])[0]

print("✅ Predicted Node Significance:", predicted_significance)
