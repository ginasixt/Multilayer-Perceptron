import os
import kagglehub
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

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
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Split data into training, validation, and test sets
# First split into train+val and test, then split train+val into train and val
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y) # 
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1765, random_state=42, stratify=y_temp)


# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

num_negative = (y_train == 0).sum().item()
num_positive = (y_train == 1).sum().item()

pos_weight = torch.tensor([num_negative / num_positive])

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

            # UPDATE: Use nn.BCEWithLogitsLoss for binary classification, no need for Sigmoid here
            # Sigmoid activation function for binary classification, so the output is a probability (0-1)
            # nn.Sigmoid() 
     )
        
    # TODO: Dropout 
      
    def forward(self, x):
        return self.model(x) 

model = MLP(input_dim=X_train.shape[1])

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4) # L2 regularization with weight_decay weight_decay=1e-4, verbessert aich minimal
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5,
)

# Lists to store losses for plotting later
train_losses = []
val_losses = []

best_val_loss = float("inf")
patience = 100
counter = 0
# Training Loop
print("[INFO] Training model ...")
for epoch in range(1000):
    model.train()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val)
    
    train_losses.append(loss.item())
    val_losses.append(val_loss.item())

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Val Loss {val_loss.item():.4f}")

     # Early stopping
    if val_loss.item() < best_val_loss:
        best_val_loss = val_loss.item()
        counter = 0
        best_model_state = model.state_dict()  # save best model
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    scheduler.step(val_loss.item())

# Load best model
model.load_state_dict(best_model_state)


# Plotting the training and validation loss
print("[INFO] Plotting training and validation loss ...")
plt.style.use('ggplot')
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss', linestyle='--')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs. Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_plot.png")
plt.show()

# Evaluation
model.eval()
with torch.no_grad():
    logits = model(X_test)                  # <- model gibt nun rohe Logits aus
    y_pred_probs = torch.sigmoid(logits)   # <- wandelt Logits in Wahrscheinlichkeiten um
    y_pred = (y_pred_probs >= 0.5).float() # <- Schwelle anwenden für binäre Vorhersage

print("\n=== Evaluation ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# AUC-ROC Berechnung
y_true = y_test.numpy()               # Falls y_test noch ein Tensor ist
y_scores = y_pred_probs.numpy()      # Modell-Output (Wahrscheinlichkeiten)

fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = roc_auc_score(y_true, y_scores)

# Plot
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], "k--")  # Diagonale
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig("ROC_curve.png")
plt.show()

