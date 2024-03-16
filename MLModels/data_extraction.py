import os
import scipy.io
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import seaborn as sns

import random_forest_classifier as rfc
import decision_tree_classifier as dtc
import xgb_classifier as xgb
import k_nearest_neighbors_classifier as knn
import ada_boost_classifier as adb
import support_vector_classifier as svc
import save_models as sm

# The directory containing the Experimental data
directory = '../data/'

welding_current_list = []
electrode_force_list = []
capacitor_voltage_list = []
microphone_voltage_list = []
microphone_time_voltage_list = []
welding_voltage_ee_list = []
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
    # print(nut_index)
    return nut_types[nut_index]


for index, filename in enumerate(os.listdir(directory), start=1):
    if filename.endswith('.mat'):

        mat_data = scipy.io.loadmat(os.path.join(directory, filename))

        file_number = int(filename.split('_')[-1].split('.')[0])

        if file_number <= 40:
            current_input = "optimal"
        else:
            current_input = "High"

        material_type = get_material_type(index)
        # print(index)

        weldingCurrent = mat_data['Data1_current_welding___Current'][:]
        electrodeForce = mat_data['Data1_force_electrode_normal'][:]
        capacitorVoltage = mat_data['Data1_voltage_capacitor_B'][:]
        microphoneVoltage = mat_data['Data1_voltage_microphone_1'][:]
        microphoneTimeVoltage = mat_data['Data1_time_voltage_microphone_1'][:]
        weldingVoltageEE = mat_data['Data1_AI_A_3_voltage_welding_ee'][:]

        # Smoothing the values
        smoothWeldingCurrent = smoothing_values(weldingCurrent)
        smoothElectrodeForce = smoothing_values(electrodeForce)
        smoothCapacitorVoltage = smoothing_values(capacitorVoltage)
        smoothMicrophoneVoltage = smoothing_values(microphoneVoltage)
        smoothMicrophoneTimeVoltage = smoothing_values(microphoneTimeVoltage)
        smoothWeldingVoltageEE = smoothing_values(weldingVoltageEE)
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
        smoothWeldingVoltageEE1 = smoothWeldingVoltageEE[startingIndex:endingIndex]

        # Append data to lists
        welding_current_list.extend(smoothWeldingCurrent1.flatten())
        electrode_force_list.extend(smoothElectrodeForce1.flatten())
        capacitor_voltage_list.extend(smoothCapacitorVoltage1.flatten())
        microphone_voltage_list.extend(smoothMicrophoneVoltage1.flatten())
        microphone_time_voltage_list.extend(smoothMicrophoneTimeVoltage1.flatten())
        welding_voltage_ee_list.extend(smoothWeldingVoltageEE1.flatten())
        current_input_list.extend([current_input] * len(smoothWeldingCurrent1.flatten()))
        material_type_list.extend([material_type] * len(smoothWeldingCurrent1.flatten()))

# Saving into DataFrame
dataExtractionAndCleaning = pd.DataFrame({
    'welding_current': welding_current_list,
    'welding_voltage_ee': welding_voltage_ee_list,
    'electrode_force': electrode_force_list,
    'capacitor_voltage': capacitor_voltage_list,
    'microphone_voltage': microphone_voltage_list,
    'microphone_time_voltage': microphone_time_voltage_list,
    'current_input': current_input_list,
    'material_type': material_type_list
})

forSeperatePrediction = pd.DataFrame({
    'welding_current': welding_current_list,
    'welding_voltage_ee': welding_voltage_ee_list,
    'electrode_force': electrode_force_list,
    'capacitor_voltage': capacitor_voltage_list,
    'microphone_voltage': microphone_voltage_list
})

# forPlotting = pd.DataFrame(
#     {'microphone_voltage': microphone_voltage_list, 'microphone_time_voltage': microphone_time_voltage_list,
#      'material_type': material_type_list})
# grouped_data = dataExtractionAndCleaning.groupby('material_type')
#
# # Define colors for each material type
# colors = {'M6 Vierkant Schweissmutter Blank': 'blue', 'Vierkant-AnschweiSSmu blank(Dark)': 'red', 'M6 Lang': 'green',
#           'M6 Zinc Coated': 'yellow', 'M6 Big Segment Buchel': 'orange'}
#
# # Plot each group separately
# plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
# for material, group in grouped_data:
#     plt.scatter(group['microphone_time_voltage'], group['microphone_voltage'], label=material, color=colors[material])
#
# # Add labels and title
# plt.xlabel('Microphone Time Voltage')
# plt.ylabel('Microphone Voltage')
# plt.title('Microphone Voltage vs Microphone Time Voltage by Material Type')
#
# # Add legend
# plt.legend()
#
# # Show plot
# plt.grid(True)
# plt.show()

# sns.set_style("whitegrid")
# plt.figure(figsize=(10, 6))
# sns.scatterplot(data=dataExtractionAndCleaning, x='welding_current', y='microphone_voltage', hue='material_type',
#                 palette='Set1', alpha=0.7)
# plt.title('Welding Current vs Microphone Voltage')
# plt.xlabel('Welding Current')
# plt.ylabel('Microphone Voltage')
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.tight_layout()
# plt.show()


# print(dataExtractionAndCleaning.shape)
# rfc.rfc_material_classification(dataExtractionAndCleaning)
# rfc.rfc_current_classification(dataExtractionAndCleaning)
# dtc.dtc_material_classification(dataExtractionAndCleaning)
# dtc.dtc_current_classification(dataExtractionAndCleaning)
# xgb.xgb_material_classification(dataExtractionAndCleaning)
# xgb.xgb_current_classification(dataExtractionAndCleaning)
# knn.knn_material_classification(dataExtractionAndCleaning)
# knn.knn_current_classification(dataExtractionAndCleaning)
# adb.adb_material_classification(dataExtractionAndCleaning)
# adb.adb_current_classification(dataExtractionAndCleaning)
# svc.svc_material_classification(dataExtractionAndCleaning)
# svc.optimized_svc_material_classification(dataExtractionAndCleaning)
# sm.rfc_material_classification(dataExtractionAndCleaning, save_model=True)

# loaded_model = joblib.load('../saveModels/rf_model.pkl')
# predictions = loaded_model.predict(forSeperatePrediction)
#
# df_predictions = pd.DataFrame(predictions, columns=['Predictions'])
#
# # Save the predictions to a CSV file
# df_predictions.to_csv('../csv/predictions.csv', index=False)
