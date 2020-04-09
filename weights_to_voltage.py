import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from scipy import interpolate


path_M1 = 'Results in CSV/linearMultiplier1.csv'
data_M1 = genfromtxt(path_M1, delimiter=',', dtype=None, invalid_raise = False)
data_M1 = data_M1[1:1002, :]



Vin_M1 = data_M1[:, 0]
print(data_M1.shape,Vin_M1.shape[0])
Iin_M1 = np.zeros((data_M1.shape[0],4))
Iin_M1_interp1= []
Iin_M1_interp2= []

# interpolate across Vin(Vgs)
interp_factor1 = 1 * Vin_M1.shape[0]
Vin_M1_interp = np.linspace(-20, 20, num = interp_factor1, endpoint = True)
for i in range(1,5):
    Iin_M1[:,i-1] = data_M1[:,i]
    interp1 = interpolate.splrep(Vin_M1, Iin_M1[:,i-1], s=0, k=2)
    Iin_M1_interp1.append(interpolate.splev(Vin_M1_interp, interp1, der=0))

#Iin_M1_interp1 = np.reshape(Iin_M1_interp1, (interp_factor1, 4))
Iin_M1_interp1 = np.array(Iin_M1_interp1).T
print(Iin_M1_interp1.shape)


# interpolate across V2
interp_factor2 = 5 * Iin_M1_interp1.shape[1]+1
V2 = np.linspace(-20, -5, num = interp_factor2, endpoint = True)
for i in range(0,interp_factor1):
    interp2 = interpolate.splrep([-20, -15, -10, -5], np.flip(Iin_M1_interp1[i,:]), s=0, k=2)
    Iin_M1_interp2.append(interpolate.splev(np.flip(V2), interp2, der=0))

Iin_M1_interp2 = np.array(Iin_M1_interp2)
print(Iin_M1_interp2[0,:])



#Plot the Data
for i in range(0,interp_factor2):
    plt.plot(Vin_M1_interp, Iin_M1_interp2 [:,i], label='V2 = %s ' % -(i))

plt.ylabel('I')
plt.xlabel('Vgs_M1')
plt.title("I-V plot for the M1 OTFT")
plt.legend()
plt.show()