# %% IMPORTS

import numpy as np
import matplotlib.pyplot as plt

# %% LOAD

patient4 = np.load("/home/jhedemann/slow-wave//1024hz/Patient04_electrode01.npy")
# patient4_mrk = np.load("/home/jhedemann/slow-wave/Patient02_OfflineMrk.mrk")

with open("/home/jhedemann/slow-wave/1024hz/Patient04_OfflineMrk.mrk", "r", encoding="utf-8") as input_file:
    text = input_file.read()



# %% DO STUFF WITH LOADED DATA

#print(patient4)
print(patient4.shape)
lines = text.splitlines()[1:]  # skip header
patient4_sw = np.array([int(line.split()[0]) / 1024 for line in lines if line.strip()])

x_patient4 = np.arange(len(patient4))/1024

# %%

plt.figure(figsize=(12,3))
plt.vlines(patient4_sw, -7500, 7500, colors="red")
plt.plot(x_patient4, patient4)


plt.xlabel("time, seconds")
plt.ylabel("EEG, microvolts")
plt.show()

# %%
