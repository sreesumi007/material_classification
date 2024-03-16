import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def knn_material_classification(data):
    label_encoder = LabelEncoder()
    data['current_input'] = label_encoder.fit_transform(data['current_input'])
    data['material_type'] = label_encoder.fit_transform(data['material_type'])

    # Define features and target variable
    X = data.drop(
        ['current_input', 'material_type', 'microphone_time_voltage'], axis=1)
    y = data['material_type']

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize and train KNeighborsClassifier
    knn_classifier = KNeighborsClassifier(n_neighbors=5)
    knn_classifier.fit(X_train_scaled, y_train)

    # Predict on the test set
    y_pred = knn_classifier.predict(X_test_scaled)

    # Evaluate the classifier
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
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues", xticklabels=np.unique(y),
                yticklabels=np.unique(y))
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix for KNeighborsClassifier')
    plt.show()


def knn_current_classification(data):
    label_encoder = LabelEncoder()
    data['current_input'] = label_encoder.fit_transform(data['current_input'])
    data['material_type'] = label_encoder.fit_transform(data['material_type'])

    # Define features and target variable
    X = data.drop(
        ['current_input', 'material_type', 'microphone_time_voltage'], axis=1)
    y = data['current_input']

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize and train KNeighborsClassifier
    knn_classifier = KNeighborsClassifier(n_neighbors=5)
    knn_classifier.fit(X_train_scaled, y_train)

    # Predict on the test set
    y_pred = knn_classifier.predict(X_test_scaled)

    # Evaluate the classifier
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
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues", xticklabels=np.unique(y),
                yticklabels=np.unique(y))
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix for KNeighborsClassifier')
    plt.show()
