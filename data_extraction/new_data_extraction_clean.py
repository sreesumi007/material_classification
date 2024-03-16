import os
import scipy.io
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Specify the directory containing the MATLAB files
directory = '../data/'

# Initialize lists to store data from each file
welding_current_list = []
electrode_force_list = []
capacitor_voltage_list = []
microphone_voltage_list = []
microphone_time_voltage_list = []
sample_rate_list = []
current_input_list = []
material_type_list = []


def smoothing_values(a, n=10):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def get_material_type(file_index):
    nut_types = ["M6 Vierkant Schweissmutter Blank", "Vierkant-AnschweiSSmu blank(Dark)", "M6 Lang", "M6 Zinc Coated",
                 "M6 Big Segment Buchel"]
    nut_index = (file_index - 1) // 60  # Each nut type has 60 files
    return nut_types[nut_index]


# Loop through files in the directory
for index, filename in enumerate(os.listdir(directory), start=1):
    if filename.endswith('.mat'):  # Check if the file is a MATLAB file
        # Load data from the MATLAB file
        mat_data = scipy.io.loadmat(os.path.join(directory, filename))

        file_number = int(filename.split('_')[-1].split('.')[0])

        if file_number <= 40:
            current_input = "optimal"
        else:
            current_input = "High"

        material_type = get_material_type(index)

        weldingCurrent = mat_data['Data1_current_welding___Current'][:]
        electrodeForce = mat_data['Data1_force_electrode_normal'][:]
        capacitorVoltage = mat_data['Data1_voltage_capacitor_B'][:]
        microphoneVoltage = mat_data['Data1_voltage_microphone_1'][:]
        microphoneTimeVoltage = mat_data['Data1_time_voltage_microphone_1'][:]

        # Smoothing the values
        smoothWeldingCurrent = smoothing_values(weldingCurrent)
        smoothElectrodeForce = smoothing_values(electrodeForce)
        smoothCapacitorVoltage = smoothing_values(capacitorVoltage)
        smoothMicrophoneVoltage = smoothing_values(microphoneVoltage)
        smoothMicrophoneTimeVoltage = smoothing_values(microphoneTimeVoltage)
        sampleRate = np.array(mat_data['Sample_rate'])[0]

        currentRiseStarts = np.argmax(smoothWeldingCurrent > 1)
        preTime = 0.0005
        postTime = 0.005

        startingIndex = int(currentRiseStarts - (sampleRate * preTime))
        endingIndex = int(currentRiseStarts + (sampleRate * postTime))
        smoothMicTime = microphoneTimeVoltage[0][startingIndex:endingIndex]
        triggerTime = currentRiseStarts * (1 / sampleRate)

        # Extracting the data in the required time frame
        smoothWeldingCurrent1 = smoothWeldingCurrent[startingIndex:endingIndex]
        smoothElectrodeForce1 = smoothElectrodeForce[startingIndex:endingIndex]
        smoothCapacitorVoltage1 = smoothCapacitorVoltage[startingIndex:endingIndex]
        smoothMicrophoneVoltage1 = smoothMicrophoneVoltage[startingIndex:endingIndex]
        smoothMicrophoneTimeVoltage1 = smoothMicrophoneTimeVoltage[startingIndex:endingIndex]

        # Append data to lists
        welding_current_list.extend(smoothWeldingCurrent1.flatten())
        electrode_force_list.extend(smoothElectrodeForce1.flatten())
        capacitor_voltage_list.extend(smoothCapacitorVoltage1.flatten())
        microphone_voltage_list.extend(smoothMicrophoneVoltage1.flatten())
        microphone_time_voltage_list.extend(smoothMicrophoneTimeVoltage1.flatten())
        current_input_list.extend([current_input] * len(smoothWeldingCurrent1.flatten()))
        material_type_list.extend([material_type] * len(smoothWeldingCurrent1.flatten()))

# Provided DataFrame
originalDatafFetch = pd.DataFrame({
    'welding_current': welding_current_list,
    'electrode_force': electrode_force_list,
    'capacitor_voltage': capacitor_voltage_list,
    'microphone_voltage': microphone_voltage_list,
    'microphone_time_voltage': microphone_time_voltage_list,
    'current_input': current_input_list,
    'material_type': material_type_list
})

# print(originalDatafFetch)

# RFC
# Separate features and target variable
X = originalDatafFetch.drop(['current_input', 'material_type'], axis=1)
# y = originalDatafFetch.drop(
#     ['welding_current', 'electrode_force', 'capacitor_voltage', 'microphone_voltage', 'microphone_time_voltage'],
#     axis=1)

y_current_input = originalDatafFetch['current_input']
y_material_type = originalDatafFetch['material_type']

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_current_input, test_size=0.2, random_state=42)

# Initialize and train the random forest classifier
rf_classifier = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
rf_classifier.fit(X_train, y_train)

# Predictions
y_pred = rf_classifier.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
