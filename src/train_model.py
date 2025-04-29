import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import joblib  # new
import pickle

def train_and_evaluate(features_path: str, defaults_path: str, save_model_path: str, save_scaler_path: str):
    data = pd.read_pickle(defaults_path).merge(
        pd.read_pickle(features_path), on='client_id', how='left'
    ).fillna(0)

    X = data.drop(columns=['client_id','default'])
    y = data['default']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    rf = RandomForestClassifier(class_weight='balanced', random_state=42)
    grid = GridSearchCV(rf, {'n_estimators':[100,200]}, scoring='roc_auc', cv=3)
    grid.fit(X_train_s, y_train)
    best = grid.best_estimator_

    y_pred = best.predict(X_test_s)
    y_proba = best.predict_proba(X_test_s)[:,1]
    print("Best params:", grid.best_params_)
    print("Test AUC:", roc_auc_score(y_test,y_proba))
    print(classification_report(y_test,y_pred))

    # ✅ Save model and scaler
    with open(save_model_path, 'wb') as f:
        pickle.dump(best, f)
    with open(save_scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train model with features.")
    parser.add_argument("--features", required=True)
    parser.add_argument("--defaults", required=True)
    parser.add_argument("--model_out", required=True)
    parser.add_argument("--scaler_out", required=True)
    args = parser.parse_args()

    train_and_evaluate(args.features, args.defaults, args.model_out, args.scaler_out)
