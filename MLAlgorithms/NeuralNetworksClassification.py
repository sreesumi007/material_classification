import os
import scipy.io
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import confusion_matrix

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
    nut_index = (file_index - 1) // 60  # Each nut type has 60 experiments
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

# Neural Network Model
# Separate features and target variable
X = originalDatafFetch.drop(
    ['current_input', 'material_type', 'microphone_time_voltage'], axis=1)
y_material_type = pd.get_dummies(originalDatafFetch['material_type'])  # One-hot encode the target variable

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_material_type, test_size=0.2, random_state=42)

# Define the neural network model
model = Sequential([
    Dense(256, input_shape=(X_train.shape[1],), activation='relu'),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(5, activation='softmax')  # 5 classes for material types
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print('Accuracy:', accuracy)

# Confusion Matrix
y_pred = model.predict(X_test)
conf_matrix = confusion_matrix(np.argmax(y_test.values, axis=1), np.argmax(y_pred, axis=1))
print('Confusion Matrix:')
print(conf_matrix)
