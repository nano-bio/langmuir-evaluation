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
    # copy from temporary data, but delete first column
    data[:, :, i - 1] = np.delete(tempdata[np.where(tempdata[:, 0] == i)], 0, 1)

# set tempdata to None so the GC can free memory
tempdata = None

# calculate plasma potential for each measurement
# preallocate array for the plasma potentials
vp = np.zeros((nom, 1))

# go through all measurements
for i in range(0, nom-1):
    # not an untrusted measurement
    error = False

    # first smooth the data and take the first derivative
    ea = np.gradient(savgol_filter(data[:, 1, i], 11, 2))
    # second derivative and smooth this again
    za = savgol_filter(np.gradient(ea), 21, 2)

    # this typically has a maximum, followed by a zero-crossing and a minimum
    # we search for the zero-crossing between the global maximum and minimum
    maximum = data[np.where(za == np.max(za)), 0, i]
    minimum = data[np.where(za == np.min(za)), 0, i]

    # get the data points between max and min
    fitrangex = data[np.where((data[:, 0, i] >= maximum) & (data[:, 0, i] <= minimum)), 0, i][1]
    fitrangey = za[np.where((data[:, 0, i] >= maximum) & (data[:, 0, i] <= minimum))[1]]

    # sometimes we hit the wrong interval. don't throw errors in that case and mark the measurement as untrusted
    try:
        # fit the range between max and min with a third order polynomial
        linfit = np.poly1d(np.polyfit(fitrangex, fitrangey, 3))
        linfitx = np.linspace(maximum[0], minimum[0], 100)
    except:
        print('Fehler in Iteration {}'.format(i))
        error = True

    # if we didn't mark it as untrusted, we take the zero-crossing
    if error:
        vp[i] = None
    else:
        # if trusted, we take the zero crossing between max and min.
        for zerocrossing in linfit.r:
            if (zerocrossing > maximum) & (zerocrossing < minimum):
                vp[i] = zerocrossing

print(vp)
#plt.plot(linfitx, linfit(linfitx))
#plt.plot(data[:, 0, 40], data[:, 1, 40])
#plt.plot(data[:, 0, 40], za)
#plt.show()
