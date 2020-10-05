import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from scipy import interpolate



def Cin_convert():
    path_M1 = 'Results in CSV/linearMultiplier2.csv'
    data_M1 = genfromtxt(path_M1, delimiter=',', dtype=None, invalid_raise = False)
    data_M1 = data_M1[326:446, :]
    # 1:446 --> all
    #326:446 --> -7--> -2.2
    #301:501  --> -8 -->0



    Vin_M1 = data_M1[:, 0]
    #print(data_M1.shape,Vin_M1.shape[0])
    Iin_M1 = np.zeros((data_M1.shape[0],4))
    Iin_M1_interp1= []
    Iin_M1_interp2= []
    p = []
    q = []

    # interpolate across Vin(Vgs)
    interp_factor1 = 1* Vin_M1.shape[0]+1
    Vin_M1_interp = np.linspace(-7, -2.2, num = interp_factor1, endpoint = True)
    for i in range(1,5):
        Iin_M1[:,i-1] = data_M1[:,i]
        interp1 = interpolate.splrep(Vin_M1, Iin_M1[:,i-1], s=0, k=1)
        Iin_M1_interp1.append(interpolate.splev(Vin_M1_interp, interp1, der=0))

    Iin_M1_interp1 = np.array(Iin_M1_interp1).T
    #print(Vin_M1_interp)


    # interpolate across V2
    interp_factor2 = 10 * Iin_M1_interp1.shape[1]+1
    V2 = np.linspace(-20, -0, num = interp_factor2, endpoint = True)
    for i in range(0,interp_factor1):
        interp2 = interpolate.splrep([-20, -15, -10, -5], np.flip(Iin_M1_interp1[i,:]), s=1, k=1)
        Iin_M1_interp2.append(interpolate.splev(np.flip(V2), interp2, der=0))


    Iin_M1_interp2 = np.array(Iin_M1_interp2)
    #print(Iin_M1_interp2[0,:])

    Iin_M1_interp2 = -Iin_M1_interp2
    Iin_M1_interp2 = np.flip(Iin_M1_interp2,axis=0)
    Vin_M1_interp = -Vin_M1_interp
    Vin_M1_interp = np.flip(Vin_M1_interp)
    V2 = -V2
    V2 = np.flip(V2)


    for i in range(interp_factor2):
        p.append(np.polyfit(Vin_M1_interp, Iin_M1_interp2[:,i], 1))
        # plt.plot(Vin_M1_interp, Iin_M1_interp2[:, i])
        # plt.plot(Vin_M1_interp, Vin_M1_interp*p[i][0]+p[i][1])

    for i in range(interp_factor1):
        q.append(np.polyfit(V2, Iin_M1_interp2[i,:], 1))
        # plt.plot(V2, Iin_M1_interp2[i, :])
        # plt.plot(V2, V2*q[i][0]+q[i][1])

    p = np.array(p)
    q = np.array(q)
    #Plot the Data
    # for i in range(0,interp_factor2):
    #     plt.plot(Vin_M1_interp, Iin_M1_interp2 [:,i], label='V2 = %s ' % V2[i])
    # V1 = Vin_M1_interp  -2
    # plt.plot(V1, Iin_M1_interp2[:, 20])


    return Iin_M1_interp2,p, Vin_M1_interp, V2

if __name__ == "__main__":
    Iin_M1_interp2,p, Vin_M1_interp, V2 = Cin_convert()
    plt.ylabel('I')
    plt.xlabel('Vgs_sensor')
    plt.title("I-V plot for the M1 OTFT")
    plt.legend()
    print(p.shape)
    plt.show()