import os
import joblib
import pandas as pd
import scipy.sparse as sp

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix
)


# ============================================================
# 1. LOAD PROCESSED DATA
# ============================================================

DATA_DIR = "../data/processed"

X_train = sp.load_npz(
    os.path.join(DATA_DIR, "X_train_tfidf.npz")
)

X_test = sp.load_npz(
    os.path.join(DATA_DIR, "X_test_tfidf.npz")
)

y_train = pd.read_csv(
    os.path.join(DATA_DIR, "y_train.csv")
)["label"]

y_test = pd.read_csv(
    os.path.join(DATA_DIR, "y_test.csv")
)["label"]


# ============================================================
# 2. BASELINE MODEL
# ============================================================

baseline = DummyClassifier(
    strategy="most_frequent"
)

baseline.fit(X_train, y_train)

baseline_pred = baseline.predict(X_test)

print("Baseline Accuracy:",
      accuracy_score(y_test, baseline_pred))


# ============================================================
# 3. CROSS-VALIDATION SETUP
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ============================================================
# 4. DEFINE CANDIDATE MODELS
# ============================================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "Multinomial Naive Bayes": MultinomialNB(),

    "Linear SVM": LinearSVC(
        random_state=42
    )
}


# ============================================================
# 5. HYPERPARAMETER TUNING
# ============================================================

# Logistic Regression
log_param_grid = {
    "C": [0.01, 0.1, 1, 10, 100]
}

log_grid = GridSearchCV(
    LogisticRegression(
        max_iter=1000,
        random_state=42
    ),
    log_param_grid,
    scoring="average_precision",
    cv=cv,
    n_jobs=-1
)

log_grid.fit(X_train, y_train)


# Multinomial Naive Bayes
nb_param_grid = {
    "alpha": [0.01, 0.1, 0.5, 1, 2]
}

nb_grid = GridSearchCV(
    MultinomialNB(),
    nb_param_grid,
    scoring="average_precision",
    cv=cv,
    n_jobs=-1
)

nb_grid.fit(X_train, y_train)


# Linear SVM
svm_param_grid = {
    "C": [0.01, 0.1, 1, 10, 100]
}

svm_grid = GridSearchCV(
    LinearSVC(random_state=42),
    svm_param_grid,
    scoring="average_precision",
    cv=cv,
    n_jobs=-1
)

svm_grid.fit(X_train, y_train)


# ============================================================
# 6. DISPLAY BEST PARAMETERS AND CV SCORES
# ============================================================

print("\nLogistic Regression:")
print("Best Parameters:", log_grid.best_params_)
print("Best CV PR-AUC:", log_grid.best_score_)

print("\nMultinomial Naive Bayes:")
print("Best Parameters:", nb_grid.best_params_)
print("Best CV PR-AUC:", nb_grid.best_score_)

print("\nLinear SVM:")
print("Best Parameters:", svm_grid.best_params_)
print("Best CV PR-AUC:", svm_grid.best_score_)


# ============================================================
# 7. MODEL COMPARISON
# ============================================================

comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Multinomial Naive Bayes",
        "Linear SVM"
    ],

    "CV PR-AUC": [
        log_grid.best_score_,
        nb_grid.best_score_,
        svm_grid.best_score_
    ]
})

print("\nModel Comparison:")
print(comparison)


# ============================================================
# 8. SELECT FINAL MODEL
# ============================================================

final_model = nb_grid.best_estimator_

print("\nFinal Model: Multinomial Naive Bayes")


# ============================================================
# 9. TEST FINAL MODEL
# ============================================================

final_model.fit(X_train, y_train)

test_pred = final_model.predict(X_test)


# ============================================================
# 10. EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test,
    test_pred
)

precision = precision_score(
    y_test,
    test_pred,
    pos_label=1,
    zero_division=0
)

recall = recall_score(
    y_test,
    test_pred,
    pos_label=1,
    zero_division=0
)

f1 = f1_score(
    y_test,
    test_pred,
    pos_label=1,
    zero_division=0
)

y_test_proba = final_model.predict_proba(
    X_test
)[:, 1]

pr_auc = average_precision_score(
    y_test,
    y_test_proba
)


print("\nFinal Model Results")
print("-------------------")
print("Accuracy:", accuracy)
print("Spam Precision:", precision)
print("Spam Recall:", recall)
print("Spam F1:", f1)
print("PR-AUC:", pr_auc)


# ============================================================
# 11. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    test_pred
)

tn, fp, fn, tp = cm.ravel()

print("\nConfusion Matrix:")
print(cm)

print("True Negative :", tn)
print("False Positive:", fp)
print("False Negative:", fn)
print("True Positive :", tp)


# ============================================================
# 12. SAVE FINAL RESULTS
# ============================================================

final_results = {
    "Accuracy": accuracy,
    "Spam Precision": precision,
    "Spam Recall": recall,
    "Spam F1": f1,
    "PR-AUC": pr_auc
}

final_results_df = pd.DataFrame(
    [final_results]
)

os.makedirs("../outputs", exist_ok=True)

final_results_df.to_csv(
    "../outputs/final_model_results.csv",
    index=False
)


# ============================================================
# 13. SAVE MODEL
# ============================================================

os.makedirs("../models", exist_ok=True)

joblib.dump(
    final_model,
    "../models/spam_classifier.pkl"
)

print("\nModel saved successfully.")