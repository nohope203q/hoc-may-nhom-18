import numpy as np
import pandas as pd
import pickle
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from ensemble_models import (
    RandomForest,
    VotingClassifier,
    XGBoostCustom,
    AdaBoostMulticlass,
    tinh_accuracy
)

print("TRAINING ENSEMBLE MODELS")
# 1. LOAD VÀ CHIA DỮ LIỆU
print("\nBƯỚC 1: Load dữ liệu Iris...")
iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# 2. TRAIN RANDOM FOREST (BAGGING)
print("\nBƯỚC 2: Training Random Forest (Bagging)")
print("Đang train 20 decision trees")
rf_model = RandomForest(n_trees=20, max_depth=5)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
rf_acc = tinh_accuracy(y_test, rf_preds)
print(f"Random Forest Accuracy: {rf_acc*100:.2f}%")
with open('model_rf_custom.pkl', 'wb') as f:
    pickle.dump(rf_model, f)
print("Đã lưu: model_rf_custom.pkl")

# 3. TRAIN VOTING CLASSIFIER
print("\nBƯỚC 3: Training Voting Classifier")
voting_model = VotingClassifier()
voting_model.fit(X_train, y_train)
voting_preds = voting_model.predict(X_test)
voting_acc = tinh_accuracy(y_test, voting_preds)
print(f"Voting Classifier Accuracy: {voting_acc*100:.2f}%")
with open('model_voting_custom.pkl', 'wb') as f:
    pickle.dump(voting_model, f)
print("Đã lưu: model_voting_custom.pkl")

# 4. TRAIN XGBOOST CUSTOM (GRADIENT BOOSTING)
print("\nBƯỚC 4: Training XGBoost Custom (Boosting)")
print("Đang train 50 sequential trees")
xgb_model = XGBoostCustom(n_estimators=50, learning_rate=0.1, max_depth=3)
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)
xgb_acc = tinh_accuracy(y_test, xgb_preds)
print(f"XGBoost Custom Accuracy: {xgb_acc*100:.2f}%")
with open('model_xgboost_custom.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)
print("Đã lưu: model_xgboost_custom.pkl")

# 5. TRAIN ADABOOST (MULTICLASS)
print("\nBƯỚC 5: Training AdaBoost (Multiclass)")
print(" Đang train với One-vs-Rest strategy")
ada_model = AdaBoostMulticlass(n_clf=50)
ada_model.fit(X_train, y_train)
ada_preds = ada_model.predict(X_test)
ada_acc = tinh_accuracy(y_test, ada_preds)
print(f"AdaBoost Multiclass Accuracy: {ada_acc*100:.2f}%")
with open('model_adaboost_custom.pkl', 'wb') as f:
    pickle.dump(ada_model, f)
print("Đã lưu: model_adaboost_custom.pkl")

# 5. SO SÁNH KẾT QUẢ
print("KẾT QUẢ SO SÁNH")
results = {
    'Model': [
        'Random Forest (Bagging)',
        'Voting Classifier',
        'XGBoost Custom (Boosting)',
        'AdaBoost (Boosting)'  
    ],
    'Accuracy': [
        rf_acc,
        voting_acc,
        xgb_acc,
        ada_acc  
    ],
    'Accuracy (%)': [
        f"{rf_acc*100:.2f}%",
        f"{voting_acc*100:.2f}%",
        f"{xgb_acc*100:.2f}%",
        f"{ada_acc*100:.2f}%" 
    ]
}
df_results = pd.DataFrame(results)
print("\n" + df_results.to_string(index=False))
df_results.to_csv('model_comparison_nhom.csv', index=False)
print("\nĐã lưu kết quả: model_comparison_nhom.csv")

# 6. CHI TIẾT METRICS
print("CHI TIẾT METRICS")

def calculate_metrics(y_true, y_pred, model_name):
    n_classes = 3
    print(f"\n{model_name}:")
    print("-" * 40)
    for cls in range(n_classes):
        true_pos = np.sum((y_true == cls) & (y_pred == cls))
        false_pos = np.sum((y_true != cls) & (y_pred == cls))
        false_neg = np.sum((y_true == cls) & (y_pred != cls))
        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        class_names = ['Setosa', 'Versicolor', 'Virginica']
        print(f"  {class_names[cls]:12} - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {recall:.4f}")
calculate_metrics(y_test, rf_preds, "Random Forest")
calculate_metrics(y_test, voting_preds, "Voting Classifier")
calculate_metrics(y_test, xgb_preds, "XGBoost Custom")
calculate_metrics(y_test, ada_preds, "AdaBoost Multiclass")
print("HOÀN TẤT!")
print("\nCác file đã tạo:")
print("model_rf_custom.pkl")
print("model_voting_custom.pkl")
print("model_xgboost_custom.pkl")
print("model_adaboost_custom.pkl")  
print("model_comparison_nhom.csv")