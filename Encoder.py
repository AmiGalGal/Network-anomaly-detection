from torch import nn
import pandas as pd
import torch
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(9, 6),
            nn.ReLU(),
            nn.Linear(6, 3)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(3, 6),
            nn.ReLU(),
            nn.Linear(6, 9)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon

XTrain = pd.read_csv("SavedResults/Train.csv")
XVal = pd.read_csv("SavedResults/Val.csv")
X_train_tensor = torch.tensor(XTrain.values, dtype=torch.float32)
X_val_tensor = torch.tensor(XVal.values, dtype=torch.float32)


model = AutoEncoder()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

epochs = 30

for epoch in range(epochs):
    model.train()

    optimizer.zero_grad()

    outputs = model(X_train_tensor)
    loss = criterion(outputs, X_train_tensor)

    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

model.eval()

with torch.no_grad():
    val_recon = model(X_val_tensor)
    val_loss = torch.mean((X_val_tensor - val_recon) ** 2, dim=1)

anomaly_scores = val_loss.numpy()

threshold = np.percentile(anomaly_scores, 95)

####test
df = pd.read_csv("SavedResults/Test.csv")
df = df.drop("session_id", axis=1)
y = df["attack_detected"]
df = df.drop("attack_detected", axis=1)
df["protocol_type"] = df["protocol_type"].replace({
    "TCP": 0,
    "UDP": 1,
    "ICMP": 2,
})

df["encryption_used"] = df["encryption_used"].replace({
    "AES": 0,
    "DES": 1,
}).fillna(2)

df["browser_type"] = df["browser_type"].replace({
    "Chrome": 0,
    "Firefox": 1,
    "Safari": 2,
    "Edge": 3,
    "Unknown": 4
})

y = y.replace({
    0: False,
    1: True,
})
y = y.tolist()

scaler = joblib.load("scaler.pkl")
df = scaler.transform(df)

Xtest_tensor = torch.tensor(df, dtype=torch.float32)

model.eval()

with torch.no_grad():
    Test_recon = model(Xtest_tensor)
    test_loss = torch.mean((Xtest_tensor - Test_recon) ** 2, dim=1)

anomaly_scores = test_loss.numpy()
best_f1 = 0
best_q = 0
best_threshold = 0
for i in range(100):
    print(f"for q = {i}")
    threshold1 = np.percentile(anomaly_scores, i)
    pred = anomaly_scores > threshold1

    accuracy = accuracy_score(y, pred)
    precision = precision_score(y, pred)
    recall = recall_score(y, pred)
    f1 = f1_score(y, pred)
    if f1 > best_f1:
        best_f1 = f1
        best_q = i
        best_threshold = threshold1
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1)

print(f"best q = {best_q} with f1 = {best_f1} with threshold = {best_threshold}")
torch.save(model.state_dict(), "models/autoencoder.pth")