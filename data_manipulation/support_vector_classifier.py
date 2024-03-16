from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def svc_material_classification(data):
    X = data.drop(
        ['current_input', 'material_type'], axis=1)

    y_material_type = data['material_type']

    # Scaling
    # scaler = StandardScaler()
    # x_scaled = scaler.fit_transform(X)

    # Splitting the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_material_type, test_size=0.2, random_state=42)

    # Initializing and training the Support Vector Classifier
    svc_classifier = SVC(kernel='rbf', random_state=42)
    svc_classifier.fit(X_train, y_train)

    # Predictions
    y_pred = svc_classifier.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    f1_score_micro = f1_score(y_test, y_pred, average='micro')
    f1_score_macro = f1_score(y_test, y_pred, average='macro')
    f1_score_weighted = f1_score(y_test, y_pred, average='weighted')

    print("Accuracy SVC:", accuracy)
    print("y test shape - ", y_test.shape)
    print("f1_score_micro SVC:", f1_score_micro)
    print("f1_score_macro SVC:", f1_score_macro)
    print("f1_score_weighted SVC:", f1_score_weighted)

    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues", xticklabels=np.unique(y_material_type),
                yticklabels=np.unique(y_material_type))
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix for Support Vector Classifier')
    plt.show()


def svc_current_classification(data):
    # X = data.drop(
    #     ['current_input', 'material_type'], axis=1)
    X = data.drop(
        ['current_input', 'material_type', 'welding_current', 'capacitor_voltage', 'welding_voltage_ee'], axis=1)
    y_current_input = data['current_input']

    # Scaling
    # scaler = StandardScaler()
    # x_scaled = scaler.fit_transform(X)

    # Splitting the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_current_input, test_size=0.2, random_state=42)

    # Initializing and training the Support Vector Classifier
    svc_classifier = SVC(kernel='rbf', random_state=42)
    svc_classifier.fit(X_train, y_train)

    # Predictions
    y_pred = svc_classifier.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    f1_score_micro = f1_score(y_test, y_pred, average='micro')
    f1_score_macro = f1_score(y_test, y_pred, average='macro')
    f1_score_weighted = f1_score(y_test, y_pred, average='weighted')

    print("Accuracy SVC:", accuracy)
    print("y test shape - ", y_test.shape)
    print("f1_score_micro SVC:", f1_score_micro)
    print("f1_score_macro SVC:", f1_score_macro)
    print("f1_score_weighted SVC:", f1_score_weighted)

    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues", xticklabels=np.unique(y_current_input),
                yticklabels=np.unique(y_current_input))
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix for Support Vector Classifier')
    plt.show()


def optimized_svc_material_classification(data, sample_size=20000):
    # Sample the data if it's too large
    if len(data) > sample_size:
        data = data.sample(sample_size, random_state=42)

    X = data.drop(['current_input', 'material_type'], axis=1)
    y_material_type = data['material_type']

    # Scaling
    # scaler = StandardScaler()
    # x_scaled = scaler.fit_transform(X)

    # Splitting the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_material_type, test_size=0.2, random_state=42)

    # Initializing and training the Support Vector Classifier
    svc_classifier = SVC(kernel='rbf', random_state=42)
    svc_classifier.fit(X_train, y_train)

    # Predictions
    y_pred = svc_classifier.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    f1_score_micro = f1_score(y_test, y_pred, average='micro')
    f1_score_macro = f1_score(y_test, y_pred, average='macro')
    f1_score_weighted = f1_score(y_test, y_pred, average='weighted')

    print("Accuracy SVC:", accuracy)
    print("y test shape - ", y_test.shape)
    print("f1_score_micro SVC:", f1_score_micro)
    print("f1_score_macro SVC:", f1_score_macro)
    print("f1_score_weighted SVC:", f1_score_weighted)

    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues", xticklabels=np.unique(y_material_type),
                yticklabels=np.unique(y_material_type))
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix for Support Vector Classifier')
    plt.show()
