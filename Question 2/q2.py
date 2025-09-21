import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from xgboost import XGBClassifier
from imblearn.over_sampling import RandomOverSampler
import numpy as np

# Data Cleaning
def data_cleaning():
    outcome = pd.read_csv("Question 2/OutcomeofCasesRegisteredattheCommunityMediationCentre.csv")
    relationship = pd.read_csv("Question 2/RelationshipofPartiesinCasesRegisteredattheCommunityMediationCentre.csv")
    source = pd.read_csv("Question 2/SourceofCasesRegisteredattheCommunityMediationCentre.csv")
    registered = pd.read_csv("Question 2/RegisteredCasesattheCommunityMediationCentre.csv")

    # Merge Dataframes
    df = registered.merge(outcome, on="case_number", how="outer")
    df = df.merge(relationship, on="case_number", how="outer")
    df = df.merge(source, on="case_number", how="outer")

    # Drop NA Rows (only 1)
    df = df.dropna(subset=["type_of_dispute", "type_of_intake", "outcome_of_cases"])
    return df

# Train Model and Evaluate
def train_and_evaluate(X, y, label_encoder=None, oversample=False):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Add Random Oversampler for 'Without Settlement' Case
    if oversample:
        ros = RandomOverSampler(sampling_strategy=0.5, random_state=42)
        X_train, y_train = ros.fit_resample(X_train, y_train)

    base_model = XGBClassifier(
        eval_metric="aucpr",
        random_state=42
    )

    # Param Grid for RandomizedSearchCV Finetuning
    param_dist = {
        "n_estimators": [50, 100, 150, 200],
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.9, 1.0],
        "gamma": [0, 1, 5]
    }

    model = RandomizedSearchCV(
        base_model,
        param_distributions=param_dist,
        n_iter=150,
        scoring="f1_macro",
        cv=3,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    # Pipeline to Encode Categorical Variables
    pipeline = Pipeline(steps=[
        ('encoder', OneHotEncoder(handle_unknown='ignore')),
        ('model', model)
    ])

    # Train and Predict
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Encode Labels with Label Encoder
    if label_encoder:
        y_test = label_encoder.inverse_transform(y_test)
        y_pred = label_encoder.inverse_transform(y_pred)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, zero_division=0))

    return pipeline, X_test, y_test


if __name__ == '__main__':
    df = data_cleaning()

    # Model 1 - Proceed to Mediation or Not
    df['mediated'] = df['outcome_of_cases'].apply(
        lambda x: 'Mediated' if 'Mediation' in x else 'Not Mediated'
    )

    X1 = df[["type_of_dispute", "type_of_intake"]]
    le_1 = LabelEncoder()
    y1 = le_1.fit_transform(df['mediated'])

    print("Model 1 - Proceed to Mediation or Not")
    model_1, X_test_1, y_test_1 = train_and_evaluate(
        X1, y1, "Model 1: Mediation vs Not",
        label_encoder=le_1, tune=True
    )

    # Model 2 - With Settlement or Not
    # Filter for Mediated Cases
    df_2 = df[df['mediated'] == 'Mediated'].copy()
    df_2['settled'] = df_2['outcome_of_cases'].apply(
        lambda x: 'With Settlement' if x == 'Mediation With Settlement' else 'Without Settlement'
    )

    X2 = df_2[["type_of_dispute", "type_of_intake"]]
    le_2 = LabelEncoder()
    y2 = le_2.fit_transform(df_2['settled'])

    print("Model 2 - With Settlement or Not")
    model_2, X_test_2, y_test_2 = train_and_evaluate(
        X2, y2, "Model 2: With vs Without Settlement",
        label_encoder=le_2, oversample=True, tune=True
    )