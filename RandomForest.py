import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("data/cybersecurity.csv")
y = df["attack_detected"]
X = df.drop(["attack_detected", "session_id"], axis=1)

X["protocol_type"] = X["protocol_type"].replace({
    "TCP": 0,
    "UDP": 1,
    "ICMP": 2,
})

X["encryption_used"] = X["encryption_used"].replace({
    "AES": 0,
    "DES": 1,
}).fillna(2)

X["browser_type"] = X["browser_type"].replace({
    "Chrome": 0,
    "Firefox": 1,
    "Safari": 2,
    "Edge": 3,
    "Unknown": 4
})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))

joblib.dump(model, "models/RandomForest.pkl")
