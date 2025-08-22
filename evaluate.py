# evaluate.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("Starting model evaluation...")

# --- 1. DEFINE PATHS ---
script_dir = os.path.dirname(__file__)
DATA_PATH = os.path.normpath(os.path.join(script_dir, 'data', 'CICIDS_sample.csv'))
OUTPUT_DIR = os.path.normpath(os.path.join(script_dir, 'evaluation_results'))
os.makedirs(OUTPUT_DIR, exist_ok=True) # Create output directory if it doesn't exist

# --- 2. LOAD AND PREPARE DATA ---
try:
    df = pd.read_csv(DATA_PATH, on_bad_lines='skip')
    df.columns = df.columns.str.strip()
except FileNotFoundError:
    print(f"Error: Data file not found at {DATA_PATH}")
    exit()

# Define features and target
features_list = [
    'Destination Port', 'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
    'Total Length of Fwd Packets', 'Total Length of Bwd Packets', 'Fwd Packet Length Max',
    'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'Bwd Packet Length Max', 'Bwd Packet Length Min'
]
df_features = df[features_list].copy()
df_features.replace([np.inf, -np.inf], np.nan, inplace=True)
df_features.dropna(inplace=True)

# Create the true labels: 1 for BENIGN, -1 for Attack
# This is what we will compare our model's predictions against.
y_true = np.where(df.loc[df_features.index, 'Label'] == 'BENIGN', 1, -1)
X = df_features

# --- 3. SPLIT DATA ---
# We split the data to train on one part and test on another, unseen part.
X_train, X_test, y_train, y_test = train_test_split(X, y_true, test_size=0.3, random_state=42, stratify=y_true)

print(f"Training set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# --- 4. SCALE FEATURES ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 5. TRAIN ISOLATION FOREST MODEL ---
# The 'contamination' parameter is crucial. It's the expected percentage of anomalies.
contamination_rate = np.mean(y_train == -1)
print(f"Calculated contamination rate: {contamination_rate:.4f}")

# *** FIX: Cap the contamination rate at 0.5 as required by scikit-learn ***
# This handles cases where anomalies are more than 50% of the training data.
if contamination_rate > 0.5:
    print("Warning: Contamination rate is > 0.5. Capping at 0.5 for the model.")
    contamination_rate = 0.5

model = IsolationForest(n_estimators=100, contamination=contamination_rate, random_state=42, n_jobs=-1)
model.fit(X_train_scaled)

# --- 6. MAKE PREDICTIONS ON THE TEST SET ---
y_pred = model.predict(X_test_scaled)

# --- 7. EVALUATE AND PRINT RESULTS ---
print("\n--- Model Evaluation Report ---")
# Note: For Isolation Forest, '1' is normal (inlier) and '-1' is anomaly (outlier).
# We align our labels accordingly.
report = classification_report(y_test, y_pred, target_names=['Attack (-1)', 'Benign (1)'])
print(report)

precision = precision_score(y_test, y_pred, pos_label=-1)
recall = recall_score(y_test, y_pred, pos_label=-1)
f1 = f1_score(y_test, y_pred, pos_label=-1)

print(f"Precision (for detecting attacks): {precision:.4f}")
print(f"Recall (for detecting attacks): {recall:.4f}")
print(f"F1-Score (for detecting attacks): {f1:.4f}")

# --- 8. GENERATE AND SAVE CONFUSION MATRIX ---
conf_matrix = confusion_matrix(y_test, y_pred, labels=[-1, 1])
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted Attack', 'Predicted Benign'],
            yticklabels=['Actual Attack', 'Actual Benign'])
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')

# Save the plot
plot_path = os.path.join(OUTPUT_DIR, 'confusion_matrix.png')
plt.savefig(plot_path)
print(f"\nConfusion matrix plot saved to: {plot_path}")
plt.show()
