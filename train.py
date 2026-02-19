import numpy as np
from extract_feature import load_dataset
from sklearn.preprocessing import StandardScaler

# Load data
print("Loading training data...")
X_train, y_train = load_dataset("data/train")
print("Loading test data...")
X_test, y_test = load_dataset("data/test")

print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)