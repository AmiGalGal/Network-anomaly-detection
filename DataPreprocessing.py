import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

#Reading the Csv
df = pd.read_csv("SavedResults/TrainVal.csv")
print(df.head())
print(df.columns)
print(df.shape[0])

#keeping only the normal data
df = df[df["attack_detected"] != 1]
print(df.shape[0])

#droping unnecessary columns
df = df.drop("session_id", axis=1)
df = df.drop("attack_detected", axis=1)
print(df.columns)
print(df.head())

#replace the classes with numeric value

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

print(df.head())

#Split
train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42
)

#Normalize
scaler = StandardScaler()
train_df = pd.DataFrame(scaler.fit_transform(train_df), columns=train_df.columns)
val_df = pd.DataFrame(scaler.transform(val_df), columns=val_df.columns)
print(train_df.head())
print(val_df.head())

print(scaler)
#saving
train_df.to_csv("SavedResults/Train.csv", index=False)
val_df.to_csv("SavedResults/Val.csv", index=False)
joblib.dump(scaler, "SavedResults/scaler.pkl")
