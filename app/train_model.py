import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score

def main():
    print("Loading heart disease dataset directly from official UCI Repository...")
    
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    cols = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]
    
    # Read CSV, treating '?' as missing values
    df = pd.read_csv(url, names=cols, na_values="?")
    
    # Drop rows with missing values
    df = df.dropna()
    
    # Target in raw dataset is 0 (no disease) and 1,2,3,4 (disease present). Convert to binary 0/1.
    df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)

    X = df.drop(columns=["target"])
    y = df["target"]

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["int64", "float64"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training model...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("Model Evaluation")
    print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
    print("Recall:", round(recall_score(y_test, y_pred), 3))
    print("F1 Score:", round(f1_score(y_test, y_pred), 3))
    print("ROC AUC:", round(roc_auc_score(y_test, y_proba), 3))

    joblib.dump(model, "app/model.pkl")
    print("Model saved to app/model.pkl")

if __name__ == "__main__":
    main()
