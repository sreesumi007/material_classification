import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def moving_average(a, n=10):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    # print(f"Calculating moving average with window size {n} and input {a}.")
    return ret[n - 1:] / n


mat_data = scipy.io.loadmat('../data/Hauptversuche_5_0017.mat')
# print("Keys in the loaded MATLAB file:", mat_data.keys())

weldingCurrent = mat_data['Data1_current_welding___Current'][:]
electrodeForce = mat_data['Data1_force_electrode_normal'][:]
capacitorVoltage = mat_data['Data1_voltage_capacitor_B'][:]
microphoneVoltage = mat_data['Data1_voltage_microphone_1'][:]
microphoneTimeVoltage = mat_data['Data1_time_voltage_microphone_1'][:]
weldingVoltageEE = mat_data['Data1_AI_A_3_voltage_welding_ee'][:]
sampleRate = np.array(mat_data['Sample_rate'])[0]

forPlotting = pd.DataFrame({
    'welding_current': weldingCurrent.flatten(),
    'welding_voltage_EE': weldingVoltageEE.flatten(),
    'electrode_force': electrodeForce.flatten(),
    'capacitor_voltage': capacitorVoltage.flatten(),
    'microphone_voltage': microphoneVoltage.flatten(),
    'microphone_time_voltage': microphoneTimeVoltage.flatten()

})

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
    'Microphone_Voltage': smoothMicrophoneVoltage1
})

# Sample time series data
# microphone_voltage = np.array(checkingData.Microphone_Voltage)
#
# # Compute FFT
# fft_result = np.fft.fft(microphone_voltage)
#
# # Frequency bins
# freq_bins = np.fft.fftfreq(len(microphone_voltage))
#
# # Compute the magnitudes of the FFT coefficients (amplitudes)
# fft_magnitudes = np.abs(fft_result)
#
# # Find the corresponding frequencies
# frequencies = np.fft.fftfreq(len(microphone_voltage))
#
# # Only take the positive frequencies (since FFT of real input is symmetric)
# positive_freq_indices = frequencies > 0
# positive_frequencies = frequencies[positive_freq_indices]
# positive_fft_magnitudes = fft_magnitudes[positive_freq_indices]
#
# # Find the peak frequency
# peak_frequency = positive_frequencies[np.argmax(positive_fft_magnitudes)]
#
# print("Peak frequency:", peak_frequency)
#
# plt.figure(figsize=(10, 6))
# plt.plot(positive_frequencies, positive_fft_magnitudes)
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Magnitude')
# plt.title('FFT Spectrum of Microphone Voltage')
# plt.grid(True)
# plt.axvline(x=peak_frequency, color='r', linestyle='--', label=f'Peak Frequency: {peak_frequency:.2f} Hz')
# plt.legend()
# plt.show()
# print(smoothChecking1.microphone_voltage)

# for column in checkingData.columns:
#     # Extract the data from the column
#     data = checkingData[column].values
#
#     # Compute FFT
#     fft_result = np.fft.fft(data)
#
#     # Compute magnitudes of FFT coefficients
#     fft_magnitudes = np.abs(fft_result)
#
#     # Find corresponding frequencies
#     frequencies = np.fft.fftfreq(len(data))
#
#     # Only take the positive frequencies
#     positive_freq_indices = frequencies > 0
#     positive_frequencies = frequencies[positive_freq_indices]
#     positive_fft_magnitudes = fft_magnitudes[positive_freq_indices]
#
#     # Find the peak frequency
#     peak_frequency = positive_frequencies[np.argmax(positive_fft_magnitudes)]
#
#     # Plot FFT spectrum
#     plt.figure(figsize=(10, 6))
#     plt.plot(positive_frequencies, positive_fft_magnitudes)
#     plt.xlabel('Frequency (Hz)')
#     plt.ylabel('Magnitude')
#     plt.title(f'FFT Spectrum of {column}')
#     plt.grid(True)
#     plt.axvline(x=peak_frequency, color='r', linestyle='--', label=f'Peak Frequency: {peak_frequency:.2f} Hz')
#     plt.legend()
#     plt.show()

# Dictionary to store FFT results
# fft_data = {}
# #
# # # Compute FFT for each column
# # for column in checkingData.columns:
# #     # Extract the data from the column
# #     data = checkingData[column].values
# #
# #     # Compute FFT
# #     fft_result = np.fft.fft(data)
# #
# #     # Compute magnitudes of FFT coefficients
# #     fft_magnitudes = np.abs(fft_result)
# #
# #     # Find corresponding frequencies
# #     frequencies = np.fft.fftfreq(len(data))
# #
# #     # Only take the positive frequencies
# #     positive_freq_indices = frequencies > 0
# #     positive_frequencies = frequencies[positive_freq_indices]
# #     positive_fft_magnitudes = fft_magnitudes[positive_freq_indices]
# #
# #     # Find the peak frequency
# #     peak_frequency = positive_frequencies[np.argmax(positive_fft_magnitudes)]
# #
# #     # Store FFT data in the dictionary
# #     fft_data[column] = {
# #         'frequencies': positive_frequencies,
# #         'magnitudes': positive_fft_magnitudes,
# #         'peak_frequency': peak_frequency
# #     }
# #
# # # Print the FFT data
# # for column, data in fft_data.items():
# #     print(f'Column: {column}')
# #     print(f'Peak Frequency: {data["peak_frequency"]}')
# #     print(f'Frequencies: {data["frequencies"]}')
# #     print(f'Magnitudes: {data["magnitudes"]}')

# rms_microphone_voltage = np.sqrt(np.mean(np.square(smoothWeldingVoltageEE1)))
# print("RMS Microphone Voltage:", rms_microphone_voltage)

rms_values = {}
for column in checkingData.columns:
    # Compute RMS
    rms_values[column] = np.sqrt(np.mean(np.square(checkingData[column])))

# Create a new DataFrame for RMS values
rms_data = pd.DataFrame(rms_values, index=['RMS'])

# Transpose the DataFrame for better visualization
rms_data = rms_data.T

print(rms_data)
