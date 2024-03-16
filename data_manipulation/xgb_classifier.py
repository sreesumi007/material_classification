import numpy as np
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


def xgb_material_classification(data):
    # Separate features and target variable
    X = data.drop(
        ['current_input', 'material_type'], axis=1)
    y_material_type = data['material_type']

    # Scaling
    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(X)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_material_type, test_size=0.2, random_state=42)

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)

    # Initialize and train the XGBoost classifier
    xgb_classifier = XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    xgb_classifier.fit(X_train, y_train_encoded)

    # Predictions
    y_pred_encoded = xgb_classifier.predict(X_test)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)

    f1_score_micro = f1_score(y_test, y_pred, average='micro')
    f1_score_macro = f1_score(y_test, y_pred, average='macro')
    f1_score_weighted = f1_score(y_test, y_pred, average='weighted')

    print("Accuracy:", accuracy)
    print("f1_score_micro:", f1_score_micro)
    print("f1_score_macro:", f1_score_macro)
    print("f1_score_weighted:", f1_score_weighted)

    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues", xticklabels=np.unique(y_material_type),
                yticklabels=np.unique(y_material_type))
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix for XGBoost Classifier')
    plt.show()


def xgb_current_classification(data):
    # Separate features and target variable
    # X = data.drop(
    #     ['current_input', 'material_type'], axis=1)
    X = data.drop(
        ['current_input', 'material_type', 'welding_current', 'capacitor_voltage', 'welding_voltage_ee'], axis=1)
    y_current_input = data['current_input']

    # Scaling
    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(X)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_current_input, test_size=0.2, random_state=42)

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)

    # Initialize and train the XGBoost classifier
    xgb_classifier = XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    xgb_classifier.fit(X_train, y_train_encoded)

    # Predictions
    y_pred_encoded = xgb_classifier.predict(X_test)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)

    f1_score_micro = f1_score(y_test, y_pred, average='micro')
    f1_score_macro = f1_score(y_test, y_pred, average='macro')
    f1_score_weighted = f1_score(y_test, y_pred, average='weighted')

    print("Accuracy:", accuracy)
    print("f1_score_micro:", f1_score_micro)
    print("f1_score_macro:", f1_score_macro)
    print("f1_score_weighted:", f1_score_weighted)

    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues", xticklabels=np.unique(y_current_input),
                yticklabels=np.unique(y_current_input))
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix for XGBoost Classifier')
    plt.show()
