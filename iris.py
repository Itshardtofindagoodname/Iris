import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    cv_scores = cross_val_score(model, X_train, y_train, cv=10, scoring='accuracy')
    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)

    return conf_matrix, precision, recall, f1, cv_mean, cv_std

# K-Nearest Neighbors (KNN) model
knn = KNeighborsClassifier(n_neighbors=3)
knn_results = evaluate_model(knn, X_train, X_test, y_train, y_test)

print("KNN Results:")
print("Confusion Matrix:\n", knn_results[0])
print(f"Precision: {knn_results[1]:.2f}")
print(f"Recall: {knn_results[2]:.2f}")
print(f"F1 Score: {knn_results[3]:.2f}")
print(f"10-Fold CV Accuracy: {knn_results[4]:.2f}")
print(f"Standard Deviation of CV Accuracy: {knn_results[5]:.2f}\n")

# Support Vector Machine (SVM) model
svm = SVC(kernel='linear', random_state=42)
svm_results = evaluate_model(svm, X_train, X_test, y_train, y_test)

print("SVM Results:")
print("Confusion Matrix:\n", svm_results[0])
print(f"Precision: {svm_results[1]:.2f}")
print(f"Recall: {svm_results[2]:.2f}")
print(f"F1 Score: {svm_results[3]:.2f}")
print(f"10-Fold CV Accuracy: {svm_results[4]:.2f}")
print(f"Standard Deviation of CV Accuracy: {svm_results[5]:.2f}")