import logging
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

def train_and_eval(df, target_col='default'):
    X = df.drop(columns=[target_col,'client_id'])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # tune RF
    logging.info("Tuning Random Forest")
    rf = RandomForestClassifier(class_weight='balanced', random_state=42)
    grid = GridSearchCV(rf, {
        'n_estimators':[100,200], 'max_depth':[None,5,10]
    }, cv=3, scoring='roc_auc', n_jobs=-1)
    grid.fit(X_train_s, y_train)
    best_rf = grid.best_estimator_
    logging.info(f"RF best params: {grid.best_params_}")

    models = [
        ("LogReg", LogisticRegression(class_weight='balanced', max_iter=1000)),
        ("DecisionTree", DecisionTreeClassifier(class_weight='balanced')),
        ("RF", best_rf),
        ("XGBoost", XGBClassifier(eval_metric='auc'))
    ]

    results = {}
    for name, m in models:
        logging.info(f"Evaluating {name}")
        if name != "RF":
            m.fit(X_train_s, y_train)
        p = m.predict_proba(X_test_s)[:,1]
        a = m.predict(X_test_s)
        results[name] = {
            'auc': roc_auc_score(y_test, p),
            'accuracy': accuracy_score(y_test, a),
            'report': classification_report(y_test, a, digits=4)
        }
    return results, best_rf, scaler
