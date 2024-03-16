import os
import scipy.io
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Specify the directory containing the MATLAB files
directory = '../data/'

# Initialize lists to store data from each file
welding_current_list = []
electrode_force_list = []
capacitor_voltage_list = []
microphone_voltage_list = []
microphone_time_voltage_list = []
sample_rate_list = []


def smoothing_values(a, n=10):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


# Loop through files in the directory
for filename in os.listdir(directory):
    if filename.endswith('.mat'):  # Check if the file is a MATLAB file
        # Load data from the MATLAB file
        mat_data = scipy.io.loadmat(os.path.join(directory, filename))

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
        welding_current_list.extend([smoothWeldingCurrent1.flatten()])
        electrode_force_list.extend([smoothElectrodeForce1.flatten()])
        capacitor_voltage_list.extend([smoothCapacitorVoltage1.flatten()])
        microphone_voltage_list.extend([smoothMicrophoneVoltage1.flatten()])
        microphone_time_voltage_list.extend([smoothMicrophoneTimeVoltage1.flatten()])
        sample_rate_list.extend(sampleRate)
        # welding_current_list.extend(smoothWeldingCurrent1.flatten())
        # electrode_force_list.extend(smoothElectrodeForce1.flatten())
        # capacitor_voltage_list.extend(smoothCapacitorVoltage1.flatten())
        # microphone_voltage_list.extend(smoothMicrophoneVoltage1.flatten())
        # microphone_time_voltage_list.extend(smoothMicrophoneTimeVoltage1.flatten())
        # sample_rate_list.extend(np.repeat(sampleRate, len(weldingCurrent)).flatten())

# Create a pandas DataFrame
originalDatafFetch = pd.DataFrame({
    'welding_current': welding_current_list,
    'electrode_force': electrode_force_list,
    'capacitor_voltage': capacitor_voltage_list,
    'microphone_voltage': microphone_voltage_list,
    'microphone_time_voltage': microphone_time_voltage_list,
    'sample_rate': sample_rate_list
})

print(originalDatafFetch)

# Define Current Input
# current_input = ["optimal"] * 40 + ["high"] * 20
# num_repetitions = len(originalDatafFetch) // len(current_input)
# extended_labels = current_input * num_repetitions + current_input[:len(originalDatafFetch) % len(current_input)]
# originalDatafFetch['current_input'] = extended_labels[:len(originalDatafFetch)]
#
# # Define material type
# material_type = ["M6 Vierkant Schweissmutter Blank"] * 60 + ["Vierkant-AnschweiSSmu blank(Dark)"] * 60 + [
#     "M6 Lang"] * 60 + ["M6 Zinc Coated"] * 60 + ["M6 Big Segment Buchel"] * 60
# num_material_repetitions = len(originalDatafFetch) // len(material_type)
# extended_material_types = material_type * num_material_repetitions + material_type[
#                                                                      :len(originalDatafFetch) % len(
#                                                                          material_type)]
# originalDatafFetch['Material_Type'] = extended_material_types[:len(originalDatafFetch)]

# Random Forest Classifier
# X = originalDatafFetch.drop(['current_input', 'Material_Type'], axis=1)
# y_current_input = originalDatafFetch['current_input']
# y_material_type = originalDatafFetch['Material_Type']
#
# X_train, X_test, y_train, y_test = train_test_split(X, y_current_input, test_size=0.2, random_state=42)
#
# rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
# rf_classifier.fit(X_train, y_train)
#
# # Predictions
# y_pred = rf_classifier.predict(X_test)
#
# # Accuracy
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)

# Export to csv to check the data
# originalDatafFetch.to_csv('../csv/DataFetchClean.csv', index=False)

# Print the DataFrame or perform any further operations
# print(originalDatafFetch)
