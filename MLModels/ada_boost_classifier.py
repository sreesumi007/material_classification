import numpy as np

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def adb_material_classification(data):
    X = data.drop(
        ['current_input', 'material_type', 'microphone_time_voltage'], axis=1)
    # X = data[['microphone_voltage']]

    y_material_type = data['material_type']

    # Scaling
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(X)

    # Splitting the  data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(x_scaled, y_material_type, test_size=0.2, random_state=42)

    # Initialize and train the AdaBoost classifier
    adaboost_classifier = AdaBoostClassifier(n_estimators=50, random_state=42)
    adaboost_classifier.fit(X_train, y_train)

    # Predictions
    y_pred = adaboost_classifier.predict(X_test)

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
    plt.title('Confusion Matrix for AdaBoost Classifier with Decision Tree Base Estimator')
    plt.show()


def adb_current_classification(data):
    # Feature Extraction
    X = data.drop(
        ['current_input', 'material_type', 'microphone_time_voltage'], axis=1)
    y_current_input = data['current_input']

    # Scaling
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(X)

    # Splitting the  data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(x_scaled, y_current_input, test_size=0.2, random_state=42)

    # Initialize and train the AdaBoost classifier
    adaboost_classifier = AdaBoostClassifier(n_estimators=50, random_state=42)
    adaboost_classifier.fit(X_train, y_train)

    # Predictions
    y_pred = adaboost_classifier.predict(X_test)

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
    plt.title('Confusion Matrix for AdaBoost Classifier with Decision Tree Base Estimator')
    plt.show()
