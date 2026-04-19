# Network-anomaly-detection
Built and evaluated ML-based anomaly detection models (AE, One-Class SVM, Random Forest) for network intrusion detection.

# 🔍 Network Traffic Anomaly Detection System

A machine learning-based anomaly detection system for identifying malicious or abnormal network behavior using both supervised and unsupervised learning approaches. The project compares multiple models and demonstrates the trade-offs between precision, recall, and robustness.

---

## 🧠 Models Implemented

### 1. Autoencoder (Neural Network)
An unsupervised deep learning model that learns to reconstruct normal network traffic patterns. Anomalies are detected based on high reconstruction error.

**Results:**
- Accuracy: 0.7002  
- Precision: 0.6367  
- Recall: 0.7870  
- F1 Score: 0.7039  

**Insights:**
- Strong recall → good at detecting anomalies  
- Lower precision → more false positives  
- Useful when unseen attack patterns are expected  

---

### 2. Random Forest Classifier
A supervised ensemble model trained on labeled data to classify normal vs attack traffic.

**Results:**
- Accuracy: 0.8842  
- Precision: 0.9938  
- Recall: 0.7456  
- F1 Score: 0.8520  

**Insights:**
- Highest accuracy and precision among all models  
- Very low false positives  
- Slightly lower recall → may miss some anomalies  
- Best overall performance for known attack patterns  

---

### 3. One-Class SVM
An unsupervised kernel-based model that learns the boundary of normal data and flags deviations as anomalies.

**Results:**
- Accuracy: 0.7925  
- Precision: 0.7990  
- Recall: 0.7238  
- F1 Score: 0.7595  

**Insights:**
- Balanced performance  
- Moderate precision and recall  
- Sensitive to hyperparameter tuning (`nu`, `gamma`)  
- Performs well when normal data is well-defined  

---

### 4. Ensemble Model (Combined Approach)
A hybrid system combining predictions from Autoencoder, Random Forest, and One-Class SVM to improve robustness.

**Results:**
- Accuracy: 0.8358  
- Precision: 0.8304  
- Recall: 0.8009  
- F1 Score: 0.8154  

**Insights:**
- Best balance between precision and recall  
- More robust than individual models  
- Reduces weaknesses of single-model approaches  
- Strong generalization across different anomaly types  

---

## 📊 Overall Comparison

| Model            | Accuracy | Precision | Recall  | F1 Score |
|------------------|----------|-----------|---------|----------|
| Autoencoder      | 0.7002   | 0.6367    | 0.7870  | 0.7039   |
| Random Forest    | 0.8842   | 0.9938    | 0.7456  | 0.8520   |
| One-Class SVM    | 0.7925   | 0.7990    | 0.7238  | 0.7595   |
| Ensemble Model   | 0.8358   | 0.8304    | 0.8009  | 0.8154   |

---

## 🧪 Key Takeaways

- **Random Forest** performs best overall for labeled data scenarios  
- **Autoencoder** is most useful for detecting unseen or novel anomalies  
- **One-Class SVM** provides a balanced unsupervised baseline  
- **Ensemble approach** improves robustness and balances precision/recall trade-offs  

---

## ⚙️ Tech Stack

- Python  
- PyTorch (Autoencoder)  
- scikit-learn (SVM, Random Forest)  
- Pandas, NumPy  

---

README.md was created with GenAI

The codebase is not fully refactored or production optimized, as this project was primarily developed for exploration and experimentation.
