import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

dataset = pd.read_csv("iris.csv") # current directory

#BOBOT
def weightsum(char, weight, bias):
    w_sum = np.dot(char, weight) + bias
    return w_sum

#Fungsi Aktivasi
def activation(w_sum):

    return 1 / (1 + np.exp(-w_sum))

# Prediction
def predict(aktivasi):
    if aktivasi > 0.5:
        return 1
    else:
        return 0

# Menentukan error tiap prediksi
def error(aktivasi, type):
    return  ((type - aktivasi) * (type - aktivasi))

# DTheta
def dTtheta(char, aktivasi, type):
    dtheta = list()
    for x in range(4):
        dtheta.append(2 * char[x] * (type - aktivasi) * (1 - aktivasi) * aktivasi)
    dtheta.append(2 * (type - aktivasi) * (1 - aktivasi) * aktivasi)
    return dtheta

# Update Bobot
def update_weight(dtheta, weight):
    for x in range(4):
        weight[x] = weight[x] + (0.1 * dtheta[x])
    return weight

# Update Bias
def update_bias(dtheta, bias):
    return bias + (0.1 * dtheta[4])

# Fungsi validasi
def validate(range1, range2, weight, bias):
    char = dataset.loc[range1, "x1":"x4"]
    type = dataset.loc[range1, "type"]
    # mereset nilai sigmoid dan jumlah akurasi
    sigmoid = 0
    acc_sum = 0
    PlotError = list()
    Acc = list()
    for x in range(range1, range2):
        sigmoid = activation(weightsum(char, weight, bias))
        if predict(sigmoid) == type:
            acc_sum += 1
        type = dataset.loc[x + 1, "type"]
        char = dataset.loc[x + 1, "x1":"x4"]
    type = dataset.loc[range2, "type"]
    PlotError.append(error(sigmoid, type))
    Acc.append(acc_sum / 20)
    return [PlotError, Acc]

# Fungsi K-Fold, K=5 kali
def KFold(k1, k2, k3, k4, k5, k6):
    char = dataset.loc[0, "x1":"x4"]
    type = dataset.loc[0, "type"]
    # Inisiasi bobot dan bias
    weight = [0.45, 0.45, 0.45, 0.45]
    bias = 0.25
    PlotError_Train = list()
    Acc_Train = list()
    PlotError_Validation = list()
    Acc_Validation = list()


    # EPOCH = 300
    for epoch in range(300):
        acc_sum = 0
        sigmoid = 0
        #UNTUK TRAINING
        for x in range(k1, k3):
            sigmoid = activation(weightsum(char, weight, bias))

            if predict(sigmoid) == type:
                acc_sum += 1

            dtheta = dTtheta(char, sigmoid, type)
            weight = update_weight(dtheta, weight)
            bias = update_bias(dtheta, bias)
            type = dataset.loc[x + 1, "type"]
            char = dataset.loc[x + 1, "x1":"x4"]
        for x in range(k4, k6):
            sigmoid = activation(weightsum(char, weight, bias))

            if predict(sigmoid) == type:
                acc_sum += 1

            dtheta = dTtheta(char, sigmoid, type)
            weight = update_weight(dtheta, weight)
            bias = update_bias(dtheta, bias)
            type = dataset.loc[x + 1, "type"]
            char = dataset.loc[x + 1, "x1":"x4"]
        type = dataset.loc[k6, "type"]
        PlotError_Train.append(error(sigmoid, type))
        Acc_Train.append(acc_sum / 80) 
        validation = validate(k3, k5, weight, bias)
        PlotError_Validation.append(validation[0])
        Acc_Validation.append(validation[1])
    return [PlotError_Train, Acc_Train, PlotError_Validation, Acc_Validation]


# Range 1,2,4:data training, range 3 dan 5:validasi 
K1 = KFold(0, 39, 99, 19, 40, 79)
K2 = KFold(0, 59, 60, 79, 80, 99)
K3 = KFold(0, 19, 40, 59, 60, 99)
K4 = KFold(0, 19, 60, 39, 40, 99)
K5 = KFold(20, 59, 0, 39, 60, 99)

# Diagram Error Func.
plot_error_training = np.add(np.add(np.add(np.add(K1[0], K2[0]), K3[0]), K4[0]), K5[0]) / 5
plot_error_validation = np.add(np.add(np.add(np.add(K1[2], K2[2]), K3[2]), K4[2]), K5[2]) / 5
plt.plot(plot_error_training, label="Data Training", color='red')
plt.plot(plot_error_validation, label="Data Validasi", color='green')
plt.title('Error Chart')
plt.ylabel('Persentase Error')
plt.xlabel('Epoch')
plt.legend()
plt.savefig('myfilename.png') #simpan dalam bentuk png

# Diagram Akurasi
plot_acc_training = np.add(np.add(np.add(np.add(K1[1], K2[1]), K3[1]), K4[1]), K5[1]) / 5
plot_acc_validation = np.add(np.add(np.add(np.add(K1[3], K2[3]), K3[3]), K4[3]), K5[3]) / 5
plt.plot(plot_acc_training, label="Data Training", color='red')
plt.plot(plot_acc_validation, label="Data Validasi", color='green')
plt.title('Accuracy Diagram')
plt.ylabel('Persentase Akurasi')
plt.xlabel('Epoch')
plt.legend()
plt.savefig('myfilename1.png') #simpan dalam bentuk