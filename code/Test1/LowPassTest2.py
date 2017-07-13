import numpy as np
import matplotlib.pyplot as plt

# |All times in mS|

# Global Vars
time_step = 1
VT_array = [[],[],[]]
R = 100000
C = 4.7e-6
current_time = 0

def charge(voltage, time):
    while current_time < time:
        voltage_cap = voltage * pow(np.e,current_time/(1000*R*C))
        print(voltage_cap)
        VT_array[0].append(current_time)
        VT_array[1].append(voltage)
        VT_array[2].append(voltage_cap)
        current_time += time_step


def plot():
    print(VT_array[0])
    plt.plot(VT_array[0], VT_array[2], '-')
    plt.show()

charge(1,1000)

plot()

