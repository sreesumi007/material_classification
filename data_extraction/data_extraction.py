import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def moving_average(a, n=10):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    # print(f"Calculating moving average with window size {n} and input {a}.")
    return ret[n - 1:] / n


mat_data = scipy.io.loadmat('../data/Hauptversuche_1_0001.mat')
# print("Keys in the loaded MATLAB file:", mat_data.keys())

weldingCurrent = mat_data['Data1_current_welding___Current'][:]
electrodeForce = mat_data['Data1_force_electrode_normal'][:]
capacitorVoltage = mat_data['Data1_voltage_capacitor_B'][:]
microphoneVoltage = mat_data['Data1_voltage_microphone_1'][:]
microphoneTimeVoltage = mat_data['Data1_time_voltage_microphone_1'][:]
weldingVoltageEE = mat_data['Data1_AI_A_3_voltage_welding_ee'][:]
sampleRate = np.array(mat_data['Sample_rate'])[0]

# checkingData1 = pd.DataFrame({
#     'welding_current': weldingCurrent.flatten(),
#     'electrode_force': electrodeForce.flatten(),
#     'capacitor_voltage': capacitorVoltage.flatten(),
#     'microphone_voltage': microphoneVoltage.flatten(),
#     'microphone_time_voltage': microphoneTimeVoltage.flatten(),
#     'sample_rate': np.repeat(sampleRate, len(weldingCurrent)).flatten(),
#     'nut_type': np.repeat("Nut_One", len(weldingCurrent)).flatten()
# })

forPlotting = pd.DataFrame({
    'welding_current': weldingCurrent.flatten(),
    'welding_voltage_EE': weldingVoltageEE.flatten(),
    'electrode_force': electrodeForce.flatten(),
    'capacitor_voltage': capacitorVoltage.flatten(),
    'microphone_voltage': microphoneVoltage.flatten(),
    'microphone_time_voltage': microphoneTimeVoltage.flatten()

})

forPlotting.plot(subplots=True, figsize=(12, 8))
plt.xlabel('Time')  # You may want to adjust the x-axis label according to your data
plt.ylabel('Values')  # You may want to adjust the y-axis label according to your data
plt.title('Plot the original data')
plt.tight_layout()
plt.show()
#
# checkingData2 = pd.DataFrame({
#     'welding_current': [weldingCurrent.flatten()],
#     'electrode_force': [electrodeForce.flatten()],
#     'capacitor_voltage': [capacitorVoltage.flatten()],
#     'microphone_voltage': [microphoneVoltage.flatten()],
#     'microphone_time_voltage': [microphoneTimeVoltage.flatten()],
#     'sample_rate': [np.repeat(sampleRate, len(weldingCurrent)).flatten()]
# })

# checkingData1.to_csv('../csv/checkingData1.csv', index=False)

# print("welding current -", weldingCurrent.shape)
# print("electrode force -", electrodeForce.shape)
# print("capacitor voltage -", capacitorVoltage.shape)
# print("Microphone voltage -", microphoneVoltage.shape)
# print("Microphone Time voltage -", microphoneTimeVoltage.shape)
# print("sampleRate  -", sampleRate.shape)
#
# print("Microphone Time voltage -", microphoneTimeVoltage)

# print("End Pandas -", checkingData2)

smoothWeldingCurrent = moving_average(weldingCurrent)
smoothElectrodeForce = moving_average(electrodeForce)
smoothCapacitorVoltage = moving_average(capacitorVoltage)
smoothMicrophoneVoltage = moving_average(microphoneVoltage)
smoothMicrophoneTimeVoltage = moving_average(microphoneTimeVoltage)
smoothWeldingVoltageEE = moving_average(weldingVoltageEE)

# Time calculation for the fixing
timeOver1Index = np.argmax(smoothWeldingCurrent > 1)
preTime = 0.0005
postTime = 0.005

startingIndex = int(timeOver1Index - (sampleRate * preTime))
endingIndex = int(timeOver1Index + (sampleRate * postTime))
smoothMicTime = microphoneTimeVoltage[0][startingIndex:endingIndex]
triggerTime = timeOver1Index * (1 / sampleRate)

# Extracting the data in the required time frame
smoothWeldingCurrent1 = smoothWeldingCurrent[startingIndex:endingIndex]
smoothElectrodeForce1 = smoothElectrodeForce[startingIndex:endingIndex]
smoothCapacitorVoltage1 = smoothCapacitorVoltage[startingIndex:endingIndex]
smoothMicrophoneVoltage1 = smoothMicrophoneVoltage[startingIndex:endingIndex]
smoothMicrophoneTimeVoltage1 = smoothMicrophoneTimeVoltage[startingIndex:endingIndex]
smoothWeldingVoltageEE1 = smoothWeldingVoltageEE[startingIndex:endingIndex]

print("checking the index start - ", startingIndex)
print("checking the index end - ", endingIndex)
print('checking the shape - ',mat_data)
# print("check the columns - ", mat_data.keys())

smoothChecking1 = pd.DataFrame({
    'welding_current': [smoothWeldingCurrent1.flatten()],
    'welding_voltage_EE': [smoothWeldingVoltageEE1.flatten()],
    'electrode_force': [smoothElectrodeForce1.flatten()],
    'capacitor_voltage': [smoothCapacitorVoltage1.flatten()],
    'microphone_voltage': [smoothMicrophoneVoltage1.flatten()],
    'microphone_time_voltage': [smoothMicrophoneTimeVoltage1.flatten()],
    'sample_rate': sampleRate.flatten()
})
# print(smoothChecking1)


checkingData = pd.DataFrame({
    'Welding_Current': smoothWeldingCurrent1,
    'welding_voltage_EE': smoothWeldingVoltageEE1,
    'Electrode_Force': smoothElectrodeForce1,
    'Capacitor_Voltage': smoothCapacitorVoltage1,
    'Microphone_Voltage': smoothMicrophoneVoltage1,
    'Microphone_time_voltage': smoothMicrophoneTimeVoltage1
})

checkingData.plot(subplots=True, figsize=(12, 8))
plt.xlabel('Time')  # You may want to adjust the x-axis label according to your data
plt.ylabel('Values')  # You may want to adjust the y-axis label according to your data
plt.title('Plot the fixed time frame data')
plt.tight_layout()
plt.show()

# checkingData.to_csv('../csv/timeFrame(9483-14983).csv', index=False)

# print(checkingData1)
#
#
# processed_data_list = [{
#     'Welding_Current': smoothWeldingCurrent1,
#     'Electrode_Force': smoothElectrodeForce1,
#     'Capacitor_Voltage': smoothCapacitorVoltage1,
#     'Microphone_Voltage': smoothMicrophoneVoltage1,
#     'SampleRate': sampleRate,
#     'TriggerTime': triggerTime,
# }]
#
# Welding_Current_list = []
# Electrode_Force_list = []
# Capacitor_Voltage_list = []
# Microphone_Voltage_list = []
# SampleRate_list = []
# TriggerTime_list = []
#
# for data in processed_data_list:
#     Welding_Current_list.append(data['Welding_Current'])
#     Electrode_Force_list.append(data['Electrode_Force'])
#     Capacitor_Voltage_list.append(data['Capacitor_Voltage'])
#     Microphone_Voltage_list.append(data['Microphone_Voltage'])
#
# # Create a DataFrame
# dataFrame = pd.DataFrame({
#     'Welding_Current': Welding_Current_list,
#     'Electrode_Force': Electrode_Force_list,
#     'Capacitor_Voltage': Capacitor_Voltage_list,
#     'Microphone_Voltage': Microphone_Voltage_list,
#
# })
# dataFrame.reset_index(drop=True, inplace=True)
# dataFrame = dataFrame.apply(pd.Series.explode)
#
# # Reset index


# dataFrame = pd.DataFrame({
#     'Welding_Current': processed_data_list.Welding_Current,
#     'Electrode_Force': processed_data_list.Electrode_Force,
#     'Capacitor_Voltage': processed_data_list.Capacitor_Voltage,
#     'Microphone_Voltage': processed_data_list.Microphone_Voltage,
#     'SampleRate': processed_data_list.sampleRate,
#     'TriggerTime': processed_data_list.TriggerTime
# })
# print('Normal data - ', processed_data_list)

# data = pd.DataFrame({
#     'Welding Current': weldingCurrent.flatten(),
#     'Electrode Force': electrodeForce.flatten(),
#     'Capacitor Voltage': capacitorVoltage.flatten(),
#     'Microphone Voltage': microphoneVoltage.flatten(),
#     'Microphone Time': microphoneTimeVoltage.flatten(),
#
# })
#
# smoothData = pd.DataFrame({
#     'Welding Current': smoothWeldingCurrent.flatten(),
#     'Electrode Force': smoothElectrodeForce.flatten(),
#     'Capacitor Voltage': smoothCapacitorVoltage.flatten(),
#     'Microphone Voltage': smoothMicrophoneVoltage.flatten()
#
# })
# filtered_data = data[(data['Welding Current'] > 1) & (data['Electrode Force'] > 1) & (data['Capacitor Voltage'] > 1)]
# smooth_filtered_data = smoothData[
#     (smoothData['Welding Current'] > 1) & (smoothData['Electrode Force'] > 1) & (smoothData['Capacitor Voltage'] > 1)]
#
# only_mic = data[
#     (data['Microphone Voltage'] > 1)]

# print('Normal data - ', data)
# print('Smooth data - ', smoothData)
# print('Normal Filtered data - ', filtered_data)
# print('Smooth Filtered data data - ', smooth_filtered_data)
# print('only Mic data - ', only_mic)
