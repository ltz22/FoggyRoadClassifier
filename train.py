import numpy as np
from extract_feature import load_dataset
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Load data
# print("Loading training data...")
X_train, y_train = load_dataset("data/train")
# print("Loading test data...")
X_test, y_test = load_dataset("data/test")

print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
print("\n" + classification_report(y_test, y_pred, target_names=["clear", "foggy"]))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=["clear", "foggy"], yticklabels=["clear", "foggy"])
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump((model, scaler), f)

print("\nModel saved to model.pkl")