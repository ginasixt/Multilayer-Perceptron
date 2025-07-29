import os
import kagglehub
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Daten laden von KaggleHub
print("[INFO] Downloading dataset via kagglehub ...")
path = kagglehub.dataset_download("alexteboul/diabetes-health-indicators-dataset")
csv_path = os.path.join(path, "diabetes_binary_health_indicators_BRFSS2015.csv")
df = pd.read_csv(csv_path)
print(f"[INFO] Dataset shape: {df.shape}")

# Prepare data
X = df.drop(columns=["Diabetes_binary"]).values # Features: all columns except the target
y = df["Diabetes_binary"].values.reshape(-1, 1) # (-1: match number of rows, 1: single column for binary target)
 
# Scale features, normalize data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Define the MLP model, 
class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),  # 21 input Features (no real layer only the input data) -> 64 Neuronen
            nn.ReLU(),

            nn.Linear(64, 32), # 64 Neuronen -> 32 Neuronen
            nn.ReLU(),

            nn.Linear(32, 1), # 32 Neuronen -> 1 Ausgang (Diabetes y/n)

            # Sigmoid activation function for binary classification, so the output is a probability (0-1)
            nn.Sigmoid() 
     )
      
    def forward(self, x):
        return self.model(x)

model = MLP(input_dim=X_train.shape[1])

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
print("[INFO] Training model ...")
for epoch in range(1000):
    model.train()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Evaluation
model.eval()
with torch.no_grad():
    y_pred_probs = model(X_test)
    y_pred = (y_pred_probs >= 0.5).float()

print("\n=== Evaluation ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
