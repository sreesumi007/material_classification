import numpy as np
import pandas as pd
import joblib
import scipy
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def rfc_material_classification(data):
    X = data.drop(['current_input', 'material_type'], axis=1)
    y_material_type = data['material_type']

    # Scaling
    # scaler = StandardScaler()
    # x_scaled = scaler.fit_transform(X)

    # Splitting the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_material_type, test_size=0.2, random_state=42)

    # Initializing and training the random forest classifier
    rf_classifier = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    rf_classifier.fit(X_train, y_train)

    # Predictions
    y_pred = rf_classifier.predict(X_test)

    # New values
    new_data = new_data_fetch()
    # new_data_scaled = scaler.transform(new_data)
    predictions_rfc(rf_classifier, new_data)

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


def predictions_rfc(model, data):
    predictions = model.predict(data)
    print('new data prediction', predictions)


def moving_average(a, n=10):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    # print(f"Calculating moving average with window size {n} and input {a}.")
    return ret[n - 1:] / n


def new_data_fetch():
    mat_data = scipy.io.loadmat('../predict_data/Hauptversuche_5_0032.mat')

    weldingCurrent = mat_data['Data1_current_welding___Current'][:]
    electrodeForce = mat_data['Data1_force_electrode_normal'][:]
    capacitorVoltage = mat_data['Data1_voltage_capacitor_B'][:]
    microphoneVoltage = mat_data['Data1_voltage_microphone_1'][:]
    weldingVoltageEE = mat_data['Data1_AI_A_3_voltage_welding_ee'][:]
    sampleRate = np.array(mat_data['Sample_rate'])[0]

    smoothWeldingCurrent = moving_average(weldingCurrent)
    smoothElectrodeForce = moving_average(electrodeForce)
    smoothCapacitorVoltage = moving_average(capacitorVoltage)
    smoothMicrophoneVoltage = moving_average(microphoneVoltage)
    smoothWeldingVoltageEE = moving_average(weldingVoltageEE)

    # Time calculation for the fixing
    timeOver1Index = np.argmax(smoothWeldingCurrent > 1)
    preTime = 0.0005
    postTime = 0.005

    startingIndex = int(timeOver1Index - (sampleRate * preTime))
    endingIndex = int(timeOver1Index + (sampleRate * postTime))

    # Extracting the data in the required time frame
    smoothWeldingCurrent1 = smoothWeldingCurrent[startingIndex:endingIndex]
    smoothElectrodeForce1 = smoothElectrodeForce[startingIndex:endingIndex]
    smoothCapacitorVoltage1 = smoothCapacitorVoltage[startingIndex:endingIndex]
    smoothMicrophoneVoltage1 = smoothMicrophoneVoltage[startingIndex:endingIndex]
    smoothWeldingVoltageEE1 = smoothWeldingVoltageEE[startingIndex:endingIndex]

    rmsValues = pd.DataFrame({
        'welding_current': smoothWeldingCurrent1,
        'welding_voltage_ee': smoothWeldingVoltageEE1,
        'electrode_force': smoothElectrodeForce1,
        'capacitor_voltage': smoothCapacitorVoltage1,
        'microphone_voltage': smoothMicrophoneVoltage1
    })

    rms_values = {}
    for column in rmsValues.columns:
        # Compute RMS
        rms_values[column] = np.sqrt(np.mean(np.square(rmsValues[column])))

    # Create a new DataFrame for RMS values
    rms_data = pd.DataFrame(rms_values, index=['RMS'])
    print('rms data -', rms_data)
    # Transpose the DataFrame for better visualization
    return rms_data
