import numpy as np

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def rfc_material_classification(data, save_model=False, model_path='../saveModels/rf_model.pkl'):
    X = data.drop(['current_input', 'material_type', 'microphone_time_voltage'], axis=1)
    y_material_type = data['material_type']

    # Scaling
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(X)

    # Splitting the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(x_scaled, y_material_type, test_size=0.2, random_state=42)

    # Initializing and training the random forest classifier
    rf_classifier = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    rf_classifier.fit(X_train, y_train)

    # Save the trained model if specified
    if save_model:
        joblib.dump(rf_classifier, model_path)

    # Predictions
    y_pred = rf_classifier.predict(X_test)

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
