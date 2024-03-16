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
from sklearn.preprocessing import StandardScaler

import random_forest_classifier as rfc
import xgb_classifier as xgb
import decision_tree_classifier as dtc
import ada_boost_classifier as adb
import support_vector_classifier as svc
import k_nearest_neighbors_classifier as knn
import save_models as sm
import neural_network_classification as nnc

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

        calculateRMS = pd.DataFrame({
            'welding_current': smoothWeldingCurrent1,
            'welding_voltage_EE': smoothWeldingVoltageEE1,
            'electrode_force': smoothElectrodeForce1,
            'capacitor_voltage': smoothCapacitorVoltage1,
            'microphone_voltage': smoothMicrophoneVoltage1,
            'material_type': material_type,
            'current_input': current_input
        })

        # RMS calculation
        rms_values = {}
        for column in calculateRMS.columns[:-2]:  # Exclude the material_type and current_input columns
            # Compute RMS
            # if column == "electrode_force":
            #     rms_values[column] = np.mean(calculateRMS[column])
            # else:
            rms_values[column] = np.sqrt(np.mean(np.square(calculateRMS[column])))
            # rms_values[column] = np.mean(calculateRMS[column])

        # Add material_type and current_input to rms_values
        rms_values['material_type'] = calculateRMS['material_type'].iloc[0]
        rms_values['current_input'] = calculateRMS['current_input'].iloc[0]

        # Create a new DataFrame for RMS values
        rms_data = pd.DataFrame(rms_values, index=['RMS'])

        # Append data to lists
        welding_current_list.extend(rms_data.welding_current)
        electrode_force_list.extend(rms_data.electrode_force)
        capacitor_voltage_list.extend(rms_data.capacitor_voltage)
        microphone_voltage_list.extend(rms_data.microphone_voltage)
        welding_voltage_ee_list.extend(rms_data.welding_voltage_EE)
        current_input_list.extend(rms_data.current_input)
        material_type_list.extend(rms_data.material_type)
        # welding_current_list.extend(smoothWeldingCurrent1.flatten())
        # electrode_force_list.extend(smoothElectrodeForce1.flatten())
        # capacitor_voltage_list.extend(smoothCapacitorVoltage1.flatten())
        # microphone_voltage_list.extend(smoothMicrophoneVoltage1.flatten())
        # microphone_time_voltage_list.extend(smoothMicrophoneTimeVoltage1.flatten())
        # welding_voltage_ee_list.extend(smoothWeldingVoltageEE1.flatten())
        # current_input_list.extend([current_input] * len(smoothWeldingCurrent1.flatten()))
        # material_type_list.extend([material_type] * len(smoothWeldingCurrent1.flatten()))

# Saving into DataFrame
dataExtractionAndCleaning = pd.DataFrame({
    'welding_current': welding_current_list,
    'welding_voltage_ee': welding_voltage_ee_list,
    'electrode_force': electrode_force_list,
    'capacitor_voltage': capacitor_voltage_list,
    'microphone_voltage': microphone_voltage_list,
    'current_input': current_input_list,
    'material_type': material_type_list
})

# dataExtractionAndCleaning.to_csv('../csv/data_cleaning_new.csv', index=False)
# rfc.rfc_material_classification(dataExtractionAndCleaning)
# rfc.rfc_current_classification(dataExtractionAndCleaning)
# xgb.xgb_material_classification(dataExtractionAndCleaning)
# xgb.xgb_current_classification(dataExtractionAndCleaning)
# dtc.dtc_material_classification(dataExtractionAndCleaning)
# dtc.dtc_current_classification(dataExtractionAndCleaning)
# adb.adb_material_classification(dataExtractionAndCleaning)
# adb.adb_current_classification(dataExtractionAndCleaning)
# svc.svc_material_classification(dataExtractionAndCleaning)
# svc.svc_current_classification(dataExtractionAndCleaning)
# knn.knn_material_classification(dataExtractionAndCleaning)
# knn.knn_current_classification(dataExtractionAndCleaning)
# sm.rfc_material_classification(dataExtractionAndCleaning)
# sm.rfc_material_classification(dataExtractionAndCleaning)
# nnc.neural_network_classification(dataExtractionAndCleaning)
