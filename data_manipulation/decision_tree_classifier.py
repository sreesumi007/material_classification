import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


def dtc_material_classification(data):
    # Separate features and target variable
    X = data.drop(
        ['current_input', 'material_type'], axis=1)
    y_material_type = data['material_type']

    # Scaling
    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(X)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_material_type, test_size=0.2, random_state=42)

    # Initialize and train the decision tree classifier
    dt_classifier = DecisionTreeClassifier(random_state=42)
    dt_classifier.fit(X_train, y_train)

    # Predictions
    y_pred = dt_classifier.predict(X_test)

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
    plt.title('Confusion Matrix for Decision Tree Classifier')
    plt.show()


def dtc_current_classification(data):
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

    # Initializing and train the decision tree classifier
    dt_classifier = DecisionTreeClassifier(random_state=42)
    dt_classifier.fit(X_train, y_train)

    # Predictions
    y_pred = dt_classifier.predict(X_test)

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
    plt.title('Confusion Matrix for Decision Tree Classifier')
    plt.show()
