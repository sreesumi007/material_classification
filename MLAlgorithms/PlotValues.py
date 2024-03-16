import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Given confusion matrix - 10 epochs with one dense layer. Microphone input
# conf_matrix = np.array([[32681, 6052, 6734, 9451, 11153],
#                         [13125, 28907, 7817, 10546, 6069],
#                         [12093, 10232, 25961, 11966, 5230],
#                         [15096, 6888, 9403, 29484, 5198],
#                         [8321, 5806, 5479, 5652, 40656]])

# Given confusion matrix - 20 epochs with one dense layer. Microphone and Welding Current
# conf_matrix = np.array([[34928, 10555, 13186, 5515, 1887],
#                         [2343, 51845, 6652, 4279, 1345],
#                         [7670, 14180, 37974, 4300, 1358],
#                         [2766, 11258, 4259, 46793, 993],
#                         [3486, 6995, 3119, 3085, 49229]]
#                        )
# Given confusion matrix - 20 epochs  with two dense layer. Microphone and Welding Current
conf_matrix = np.array([[50597, 4009, 7405, 2240, 1820],
                        [5808, 50067, 5467, 3809, 1313],
                        [11034, 6874, 44098, 2348, 1128],
                        [4831, 4075, 2136, 54096, 931],
                        [3697, 2284, 974, 1540, 57419]]
                       )

# Plotting the confusion matrix
plt.figure(figsize=(10, 8))
sns.set(font_scale=1.2)  # Adjust the font scale if necessary
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', annot_kws={'size': 12})

# Adding labels
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')

# Adjusting ticks for better readability
plt.xticks(np.arange(5) + 0.5, ['Nut1', 'Nut2', 'Nut3', 'Nut4', 'Nut5'])
plt.yticks(np.arange(5) + 0.5, ['Nut1', 'Nut2', 'Nut3', 'Nut4', 'Nut5'])

plt.show()
