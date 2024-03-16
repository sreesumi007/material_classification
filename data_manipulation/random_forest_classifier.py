import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc, f1_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def rfc_material_classification(data):
    X = data.drop(
        ['current_input', 'material_type'], axis=1)
    # X = data[['microphone_voltage']]

    y_material_type = data['material_type']

    # Splitting the  data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_material_type, test_size=0.2, random_state=42)

    # Initializing and train the random forest classifier
    rf_classifier = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    rf_classifier.fit(X_train, y_train)

    # Predictions
    y_pred = rf_classifier.predict(X_test)
    # Do ROC calculation
    # roc_calculation(rf_classifier, X_test, y_test)
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    f1_score_micro = f1_score(y_test, y_pred, average='micro')
    f1_score_macro = f1_score(y_test, y_pred, average='macro')
    f1_score_weighted = f1_score(y_test, y_pred, average='weighted')

    print("Accuracy RFC:", accuracy)
    print("y test shape - ", y_test.shape)
    print("f1_score_micro RFC:", f1_score_micro)
    print("f1_score_macro RFC:", f1_score_macro)
    print("f1_score_weighted RFC:", f1_score_weighted)

    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues", xticklabels=np.unique(y_material_type),
                yticklabels=np.unique(y_material_type))
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix for Random Forest Classifier')
    plt.show()


def rfc_current_classification(data):
    # Feature Extraction
    # X = data.drop(
    #     ['current_input', 'material_type'], axis=1)
    X = data.drop(
        ['current_input', 'material_type', 'welding_current', 'capacitor_voltage', 'welding_voltage_ee'], axis=1)
    y_current_input = data['current_input']

    # Scaling
    # scaler = StandardScaler()
    # x_scaled = scaler.fit_transform(X)

    # Splitting the  data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_current_input, test_size=0.2, random_state=42)

    # Initializing and train the random forest classifier
    rf_classifier = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    rf_classifier.fit(X_train, y_train)

    # Predictions
    y_pred = rf_classifier.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    f1_score_micro = f1_score(y_test, y_pred, average='micro')
    f1_score_macro = f1_score(y_test, y_pred, average='macro')
    f1_score_weighted = f1_score(y_test, y_pred, average='weighted')

    print("Accuracy RFC:", accuracy)
    print("f1_score_micro RFC:", f1_score_micro)
    print("f1_score_macro RFC:", f1_score_macro)
    print("f1_score_weighted RFC:", f1_score_weighted)

    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues", xticklabels=np.unique(y_current_input),
                yticklabels=np.unique(y_current_input))
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix for Random Forest Classifier')
    plt.show()


def roc_calculation(model, x_test, y_test):
    y_probs = model.predict_proba(x_test)
    n_classes = y_probs.shape[1]

    # Binarize the output
    y_test_transform = label_binarize(y_test, classes=range(n_classes))

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_transform[:, i], y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curves for each class
    plt.figure()
    colors = ['blue', 'red', 'green']  # You can extend this list based on the number of classes
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve (class {0}) (area = {1:0.2f})'.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve (Multiclass)')
    plt.legend(loc="lower right")
    plt.show()
