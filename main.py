import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# lambda to convert float with , as decimal point to real point
commatodot = lambda s: float(s.decode("utf-8").replace(',', '.'))

# read textfile to temporary array using only 4 columns and the above lambda
tempdata = np.loadtxt('testdata.txt', usecols=(0, 3, 4, 5), skiprows=7,
                      converters={0: commatodot, 3: commatodot, 4: commatodot, 5: commatodot})

# number of measurements = highest value of first column
nom = int(np.amax(tempdata, axis=0)[0])

# check whether all measurements have the same number of datapoints
if tempdata.shape[0] % nom != 0:
    quit('Not an equal amount of datapoints for all measurements')

lines = int(tempdata.shape[0] / nom)

# create new array
data = np.zeros((lines, 3, nom))
for i in range(1, nom):
    data[:, :, i - 1] = np.delete(tempdata[np.where(tempdata[:, 0] == i)], 0, 1)


# unfinished from here. 
plotkurve = 11
ea = np.gradient(savgol_filter(data[:, 1, plotkurve], 11, 2))
#ea = np.gradient(data[:, 1, plotkurve])
za = savgol_filter(np.gradient(ea), 21, 2)
#za = np.gradient(ea)
plt.plot(data[:, 0, plotkurve], data[:, 1, plotkurve])
plt.plot(data[:, 0, plotkurve], za)
plt.show()
