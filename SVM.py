from sklearn.svm import OneClassSVM
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


model = OneClassSVM(
    kernel="rbf",
    nu=0.05,
    gamma="scale"
)

X_train = pd.read_csv("SavedResults/Train.csv")

model.fit(X_train)

X_val = pd.read_csv("SavedResults/Val.csv")
pred_train = model.predict(X_train)
pred_val = model.predict(X_val)

scores = model.decision_function(X_val)

anomaly_score = -scores

threshold = np.percentile(anomaly_score, 95)

pred = anomaly_score > threshold

#print(pred)




#Test
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

scoresTest = model.decision_function(df)
anomaly_scoreTest = -scoresTest


best_f1 = 0
best_q = 0
best_threshold = 0
for i in range(100):
    print(f"for q = {i}")
    thresholdTest = np.percentile(anomaly_scoreTest, i)
    predd = anomaly_scoreTest > thresholdTest

    accuracy = accuracy_score(y, predd)
    precision = precision_score(y, predd)
    recall = recall_score(y, predd)
    f1 = f1_score(y, predd)
    if f1 > best_f1:
        best_f1 = f1
        best_q = i
        best_threshold = thresholdTest
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1)

print(f"best q = {best_q} with f1 = {best_f1} with threshold = {best_threshold}")
joblib.dump(model, "models/svm_model.pkl")