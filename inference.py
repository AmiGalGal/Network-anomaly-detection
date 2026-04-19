import torch
import joblib
from torch import nn
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
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

scaler = joblib.load("SavedResults/scaler.pkl")
SVM = joblib.load("models/svm_model.pkl")

def inference_SVM(data):
    X = scaler.transform(data)
    score = SVM.decision_function(X)
    anomaly_score = -score
    threshold = -1.1593558169404248
    pred = anomaly_score > threshold
    return pred

model = AutoEncoder()
model.load_state_dict(torch.load("models/autoencoder.pth"))
model.eval()

def inference_Encoder(data):
    X = scaler.transform(data)
    threshold = 1.1031993627548218
    tensor = torch.tensor(X, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        Test_recon = model(tensor)
        test_loss = torch.mean((tensor - Test_recon) ** 2, dim=1)
    anomaly_score = test_loss.numpy()
    pred = anomaly_score > threshold
    return pred

RF = joblib.load("models/RandomForest.pkl")
def inference_RF(data):
    pred = RF.predict(data)
    return pred.astype(bool)

def inference(data):
    En = (inference_Encoder(data))
    S = (inference_SVM(data))
    R = (inference_RF(data))
    verdict = []
    for i in range(len(En)):
        print(f"Encoder: {En[i]}")
        print(f"SVM: {S[i]}")
        print(f"Random Forest: {R[i]}")
        majority = (En[i] and S[i]) or (R[i] and S[i]) or (R[i] and En[i])
        print(f"Majority: {majority}")
        verdict.append(majority)
    return verdict

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

pred = inference(df)

accuracy = accuracy_score(y, pred)
precision = precision_score(y, pred)
recall = recall_score(y, pred)
f1 = f1_score(y, pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1:", f1)
