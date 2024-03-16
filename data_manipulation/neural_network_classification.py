import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.callbacks import EarlyStopping


def neural_network_classification(data):
    # X = data.drop(
    #     ['current_input', 'material_type'], axis=1)
    X = data.drop(
        ['current_input', 'material_type', 'welding_current', 'capacitor_voltage', 'welding_voltage_ee'], axis=1)
    # y_material_type = pd.get_dummies(data['material_type'])
    y_material_type = pd.get_dummies(data['current_input'])

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_material_type, test_size=0.2, random_state=42)
    model = Sequential([
        Dense(264, activation='relu', input_shape=(X_train.shape[1],)),
        # Dense(128, activation='relu'),

        Dense(y_train.shape[1], activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=100, batch_size=10, validation_split=0.2, verbose=1)

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print('\nTest accuracy:', test_acc)
    print('\nTest loss:', test_loss)

    y_pred = model.predict(X_test)
    y_pred_categorical = np.argmax(y_pred, axis=1)
    y_test_categorical = np.argmax(y_test.values, axis=1)
    conf_matrix = confusion_matrix(y_test_categorical, y_pred_categorical)
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()
