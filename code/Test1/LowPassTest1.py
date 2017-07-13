import numpy as np

# Note: timescale in ms (milliseconds)
class LowPassSim():
    print("Instanced")
    time_voltage_array = []

    def time_setup(self, lower = 0, upper = 1000, n = 1000):
        time_array = np.linspace(lower, upper, n)
        self.time_voltage_array.append(time_array)

    def volt_setup(self, low = 0, high = 1, pulse_width = 100):
        Vin_array = []
        # 'element' corresponds to time here:
        for time in self.time_voltage_array[0]:
            if time < pulse_width:
                print("high")
                # 'element' corresponds to voltage here:
                Vin_array.append(high)
            else:
                Vin_array.append(low)
        Vin_array = np.array(Vin_array)
        self.time_voltage_array.append(Vin_array)






sim1 = LowPassSim()
sim1.time_setup()
sim1.volt_setup()

print(sim1.time_voltage_array)

