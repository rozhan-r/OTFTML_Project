
################################################################################
# otft_model.py
# version: 1.0
# Sadeghi / Justin
################################################################################
################################################################################
# This is a simulator environment for OTFT classifier algorithm development.
# For given classifier input (c_in), program gives the single voltage output for
# a given image.
# Input:  c_in (284x1) vector
# Input:  img_in(284x1) vector for MNIST
# Output:  out_volt [V]
# Example:
# otft_classifier([0,1,-0.77,1,1,0.81,0,1,-0.8,1,1,0.68,1,1,0,1,1,-0.43,1,1,1,1,
# -0.3,-1,-1,1,-1,1,0,1,0.01,1,-0.9,0,0,1])
# Please define these inputs between [-1,1] (double)
# it returns 1.45495V output
# It also plots the image to verify the entered image.
################################################################################
################################################################################

# import serial
from mnist import MNIST
import cv2
from PIL import Image
import sklearn.metrics as metrics
import sklearn.preprocessing as pp
import numpy as np
import time
import numpy as np
from pylab import *
from weights_to_voltage import Cin_convert
import csv
from collections import defaultdict
from scipy.signal import lfilter, welch, hann
from scipy import fft
import sys
import scipy.io as sio


def closest(lst, K):
    lst = np.asarray(lst)
    idx = (np.abs(lst - K)).argmin()
    return idx


def load_dataset():
    mndata = MNIST('mnist_benchmark/')
    X_train, labels_train = map(np.array, mndata.load_training())
    X_test, labels_test = map(np.array, mndata.load_testing())
    X_train = X_train/255.0
    X_test = X_test/255.0
    return (X_train, labels_train), (X_test, labels_test)


def otft_classifier(c_in, img_in):
    n = 28

    img_mat = np.reshape(img_in, (n, n))
    imshow(img_mat, interpolation='none')
    imshow(img_mat, interpolation='none', cmap='Greys_r')
    show()

    c_mat = np.reshape(c_in, (n, n))
    Vm_p = np.zeros(n * n, dtype=float)
    Vm_n = np.zeros(n * n, dtype=float)

    Id_p = np.zeros(n * n, dtype=float)
    Id_n = np.zeros(n * n, dtype=float)
    delta_Id = np.zeros(n * n, dtype=float)

    sigma_vtb = 1;
    mean_vtb = 2;
    np.random.seed(10);
    vtbn = np.random.randn(n * n);
    Vtb_vect = sigma_vtb * vtbn + mean_vtb;

    sigma_vts = 1;
    mean_vts = 2;
    np.random.seed(11);
    vtsn = np.random.randn(n * n);
    Vts_vect = sigma_vts * vtsn + mean_vts;

    sigma_kpb = 10e-9;
    mean_kpb = 100e-9;
    np.random.seed(12);
    vtbn = np.random.randn(n * n);
    Kpb_vect = sigma_kpb * vtbn + mean_kpb;

    sigma_kps = 10e-9;
    mean_kps = 30e-9;
    np.random.seed(13);
    kpsn = np.random.randn(n * n);
    Kps_vect = sigma_kps * kpsn + mean_kps;

    sigma_vtb_diff = 0.1
    sigma_vts_diff = 0.1
    sigma_kpb_diff = 1e-9
    sigma_kps_diff = 1e-9

    # global parameters circuit parameters
    Vdd = 30;  # Supply voltage, V
    R = 2e4  # Resistance, Ohm
    Av2 = 2  # Second gain satge, V/V
    Vcm = 2  # Common mode voltage, V
    Vccm = 10  # Common-mode voltage for trainer input, V
    gain_c = 4  # Gain for trainer input, V/V

    # Imager
    # First-order response, image to voltage
    Vx_mat = img_mat + Vdd / 2

    Vx_vect = np.reshape(Vx_mat, n * n)
    Vc_vect = np.reshape(c_mat, n * n)



    for k in range(0, n * n):
        Vc_p = Vccm + gain_c * Vc_vect[k]
        Vx = Vx_vect[k]
        Vtb_p = Vtb_vect[k]
        Vts_p = Vts_vect[k]
        Kpb_p = Kpb_vect[k]
        Kps_p = Kps_vect[k]

        Vc_n = Vccm - gain_c * Vc_vect[k]
        Vtb_n = Vtb_vect[k] + sigma_vtb_diff
        Vts_n = Vts_vect[k] + sigma_vts_diff
        Kpb_n = Kpb_vect[k] + sigma_kpb_diff
        Kps_n = Kps_vect[k] + sigma_kps_diff

        coeff_p = [-Kps_p / 2 - Kpb_p / 2, Kpb_p * (Vx + Vtb_p) + Kps_p * (Vc_p + Vts_p),
                   Kps_p * (Vdd - Vc_p - Vts_p) * Vdd - Vdd ** 2 / 2 * Kps_p - Kpb_p / 2 * (Vx + Vtb_p) ** 2]
        coeff_n = [-Kps_n / 2 - Kpb_n / 2, Kpb_n * (Vx + Vtb_n) + Kps_n * (Vc_n + Vts_n),
                   Kps_n * (Vdd - Vc_n - Vts_n) * Vdd - Vdd ** 2 / 2 * Kps_n - Kpb_n / 2 * (Vx + Vtb_n) ** 2]

        if (np.imag(coeff_p[0]) != 0):
            print('Degeneration device[%d] with positive polarity is NOT in triode' % k)
            break
        if (np.imag(coeff_n[0]) != 0):
            print('Degeneration device[%d] with negative polarity  is NOT in triode' % k)
            break

        root_coeff_p = roots(coeff_p)
        if (root_coeff_p[0] > (Vx + Vtb_p)):
            Vm_p[k] = root_coeff_p[0]
        else:
            Vm_p[k] = root_coeff_p[1]

        root_coeff_n = roots(coeff_n)
        if (root_coeff_n[0] > (Vx + Vtb_n)):
            Vm_n[k] = root_coeff_n[0]
        else:
            Vm_n[k] = root_coeff_n[1]

        Id_p[k] = Kpb_p / 2 * (Vm_p[k] - Vx - Vtb_p) ** 2
        Id_n[k] = Kpb_n / 2 * (Vm_n[k] - Vx - Vtb_n) ** 2

        delta_Id[k] = (Id_p[k] - Id_n[k])

    out_volt = sum(delta_Id) * R * Av2 + Vcm

    return out_volt


def otft_classifier_real(c_in, img_in, single=False):
    n = 28


    Vdd = -20;  # Supply voltage, V
    R = 2e4  # Resistance, Ohm
    Vcm = 2  # Common mode voltage, V
    Vccm = 10  # Common-mode voltage for trainer input, V

    # Imager
    # First-order response, image to voltage
    #Vx_mat = 3*img_in +3  # have input from -8 volts to -4  volts
    Vth = 0
    Vx_mat = img_in + 2
    c_in_max = np.max(np.abs(c_in))
    norm = np.linalg.norm(c_in)
    Vc_mat = c_in/norm*10
    # if c_in_max!=0.0:
    #     Vc_mat =  c_in/c_in_max*15
    # else:
    #     Vc_mat = c_in


    for i in range(Vc_mat.shape[0]):
        if Vc_mat[i]==0:
            Vc_mat[i] = Vth
        elif Vc_mat[i]> 0:
            Vc_mat[i] = Vc_mat[i]+Vth
        else:
            Vc_mat[i] = Vc_mat[i] - Vth

    _,p_coeff, Vsensor_interpolated, Vweight_interpolated = Cin_convert()
    index_weight = []

    for V_weight in Vc_mat:
        index_weight.append(closest(Vweight_interpolated, abs(V_weight)))

    Isum = zeros(1)
    I0 = zeros(1)
    if single == True:
        for item in range(len(Vx_mat)):
            ind_weight = index_weight[item]
            I0 = p_coeff[ind_weight][0] * Vx_mat[item] + p_coeff[ind_weight][1]
            if abs(Vc_mat[item]) == Vth:
                I0 = 0
            if Vc_mat[item] < 0:
                I0 = -I0
            Isum = Isum + I0

        out_volt = Isum * float(R)
        print("Hi")
        return out_volt, Vx_mat, R

    Isum = zeros(Vx_mat.shape[0])
    I0 = zeros(Vx_mat.shape[0])
    for item in range(Vx_mat.shape[1]):
        ind_weight = index_weight[item]
        I0 = p_coeff[ind_weight][0] * Vx_mat[:,item] + p_coeff[ind_weight][1]
        for j in range(I0.shape[0]):
            if abs(Vc_mat[item]) == Vth:
                I0[j] = 0
            if Vc_mat[item] < 0:
                I0[j]= -I0[j]
        Isum = Isum + I0

    #out_volt = Isum * float(R) + float(Vdd)
    #out_volt = Isum * float(R)-1.0
    out_volt = Isum * float(R)

    return out_volt,Vx_mat,R


def grad_otft(c_in, img_in):
    return

if __name__ == "__main__":

    (X_train, labels_train), (X_test, labels_test) = load_dataset()
    Image1 = X_train[0:2, :]
    img_mat = np.reshape(Image1, (2, 28, 28))
    new_image= cv2.resize(img_mat, (7, 7))

    # label1 = labels_train[0:2]
    c_in = np.zeros((Image1.shape[1],1))
    # print("Image",Image1, "\nlablel:\t", label1)
    out,_,_ = otft_classifier_real(c_in,Image1)
    # print("Output:", out.shape)







































