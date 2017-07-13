import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import time


def function(t, V0, RC):
    return V0*np.exp(-t/RC)


# determination of vars (V0 , RC)
def const_calc(point_a, point_b):   # in format point_a = [t, v]
    RC = (point_b[0] - point_a[0])/np.log(point_a[1]/point_b[1])
    V0_a = point_a[1] * np.exp(point_a[0]/RC)
    V0_b = point_b[1] * np.exp(point_b[0] / RC)
    # print("RC = ", RC, "\nV0_a = ", V0_a, "    V0_b = ", V0_b, "\ndelta: ", V0_a-V0_b)
    return RC, V0_a

def plot_graph(V0, RC):
    time_array = np.linspace(0,6*RC, 200)
    volt_array = []
    for t in time_array:
        volt_array.append(function(t, V0, RC)*1000) # mV

    plt.plot(time_array*1000,volt_array)
    plt.xlabel('time /ms')
    plt.ylabel('voltage /mv')   # CHANGE SI PREFIX
    plt.title('v/t graph at RC = ' + str('{0:.2f}'.format(RC*1000)) +' ms')

    plt.show()


R = 10          # ohms
C = 47e-6       # farads

V_in = 50e-3    # volts
tau = 100e-6    # seconds


true_RC = R*C    # Time const.
true_V0 = V_in*tau/true_RC  # safe aprox.

# points in the form of [time, voltage]
true_a = [1, function(1, true_V0, true_RC)]
true_b = [2, function(2, true_V0, true_RC)]

plot_graph(true_V0,true_RC)













