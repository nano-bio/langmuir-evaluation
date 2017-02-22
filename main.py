import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.constants import Boltzmann, elementary_charge, pi, electron_mass
from scipy import optimize

datasettoplot = 4
probe_area = 0.0001

"""
Starting from here we set up some helper functions

"""

def fromxongreaterthanzero(array):
    """
    This funtion returns an array of booleans that indicate, whether all following data points are above zero. Example:
    [0 -1 -2 -3 -2 -1 0 1 2 1 0 -1 0 1 2 3 4 5]
    returns
    [F F F F F F F F F F F T T T T T T]
    :param array:
    :return: array with booleans
    """
    returnarray = [False] * array.shape[0]
    for i in np.arange(0, array.shape[0]):
        returnarray[i] = np.greater(array[i:], 0).all()
    return returnarray

# lambda to convert float with , as decimal point to real point
commatodot = lambda s: float(s.decode("utf-8").replace(',', '.'))

# lambda that corresponds to complete current function as in http://dx.doi.org/10.1116/1.577344 eq. 1
ic_func = lambda p, x:  p[0]*np.exp(elementary_charge*(x-vp[i])/(Boltzmann*p[1]))+p[2]*np.exp(elementary_charge*(x-vp[i])/(Boltzmann*p[3]))

"""
p[0] = electron current cold
p[1] = electron temperature cold
p[2] = electron current hot
p[3] = electron temperature hot
"""

"""
From here on, the routine starts
"""

# read textfile to temporary array using only 4 columns and the above lambda
tempdata = np.loadtxt('testdata.txt', usecols=(0, 3, 4, 5), skiprows=7,
                      converters={0: commatodot, 3: commatodot, 4: commatodot, 5: commatodot}, dtype=np.float64)

# number of measurements = highest value of first column
nom = int(np.amax(tempdata, axis=0)[0])

# check whether all measurements have the same number of datapoints
if tempdata.shape[0] % nom != 0:
    quit('Not an equal amount of datapoints for all measurements')

lines = int(tempdata.shape[0] / nom)

# create new array
data = np.zeros((lines-1, 3, nom), dtype=np.float64)
tempdata2 = np.zeros((lines, 3, nom), dtype=np.float64)
for i in np.arange(0, nom):
    # copy from temporary data, but delete first column
    tempdata2[:, :, i] = np.delete(tempdata[np.where(tempdata[:, 0] == i+1)], 0, 1)

    # also delete first data point, because it is an artifact
    data[:, :, i] = np.delete(tempdata2[:,:,i], (0), axis=0)

# set tempdata to None so the GC can free memory
tempdata = None
tempdata2 = None

# prellocate some arrays / lists for data to be calculated

# preallocate array for the plasma potentials
vp = np.zeros((nom, 1), dtype=np.float64)
# list for 1st order polynomial objects for ion saturation current
ionsat = [None]*nom
# numpy array for ion saturation current subtracted data
data_is_subtracted = np.zeros((lines-1, nom), dtype=np.float64)
# preallocate array for electron temperatures (1 T_hot, 2 T_cold, 3 + 4 electron densities
temperatures = np.zeros((nom, 4), dtype=np.float64)

# go through all measurements
for i in np.arange(0, nom):
    # not an untrusted measurement
    error = False

    # first smooth the data and take the first derivative
    ea = np.gradient(savgol_filter(data[:, 1, i], 11, 2))
    # second derivative and smooth this again
    za = savgol_filter(np.gradient(ea), 21, 2)

    # this typically has a maximum, followed by a zero-crossing and a minimum
    # we search for the zero-crossing between the global maximum and minimum of the data, but not within 10% of the
    # border!
    tenpercent = int(np.round(lines/10))

    maximum = data[np.where(za == np.max(za[tenpercent:-tenpercent])), 0, i]
    minimum = data[np.where(za == np.min(za[tenpercent:-tenpercent])), 0, i]

    # get the data points between max and min
    fitrangex = data[np.where((data[:, 0, i] >= maximum) & (data[:, 0, i] <= minimum)), 0, i][1]
    fitrangey = za[np.where((data[:, 0, i] >= maximum) & (data[:, 0, i] <= minimum))[1]]

    # sometimes we hit the wrong interval. don't throw errors in that case and mark the measurement as untrusted
    try:
        # fit the range between max and min with a third order polynomial
        vpfit = np.poly1d(np.polyfit(fitrangex, fitrangey, 3))
        vpfitx = np.linspace(maximum[0], minimum[0], 100)
    except Exception as e:
        print('Fehler in Iteration {}:'.format(i))
        print(e)
        print('Minimum of second derivative: {}'.format(minimum))
        print('Maximum of second derivative: {}'.format(maximum))
        error = True

    # if we didn't mark it as untrusted, we take the zero-crossing
    if error:
        vp[i] = None
    else:
        # if trusted, we take the zero crossing between max and min.
        for zerocrossing in vpfit.r:
            if (zerocrossing > maximum) & (zerocrossing < minimum):
                vp[i] = zerocrossing

    # fit a linear function between -40 to -10 (subtract ion saturation current)

    # get the data points between max and min
    fitrangex = data[np.where((data[:, 0, i] >= -40) & (data[:, 0, i] <= -10)), 0, i][0]
    fitrangey = data[np.where((data[:, 0, i] >= -40) & (data[:, 0, i] <= -10)), 1, i][0]
    ionsat[i] = np.poly1d(np.polyfit(fitrangex, fitrangey, 1))
    ionsatx = np.linspace(-40, -10, 100)

    # create a new dataset, where we substract the ion saturation current
    def subtract_ion_sat_current(a):
        return data[np.where(data[:,0,i] == a), 1, i]-ionsat[i](a)

    data_is_subtracted[:, i] = np.apply_along_axis(subtract_ion_sat_current, 0, data[:, 0, i])

    # starting from here we fit electron temperatures and currents
    errfunc = lambda p, x, y: ic_func(p, x) - y

    # starting values
    p = [0]*4
    p[0] = 1
    p[1] = 29000 # 2.5 eV in K
    p[2] = 1
    p[3] = 145100 # 12.5 eV in K

    # select the data points to be fitted. three conditions:
    # 1) ion saturation corrected current has to be above zero (above floating potential)
    # 2) only datapoints below the plasma potential
    # 3) using the above function fromxongreaterthanzero we avoid datapoints above zero which happened due to the
    # correction

    x = data[np.where((data_is_subtracted[:, i] >= 0) & (data[:, 0, i] <= vp[i]) & fromxongreaterthanzero(data_is_subtracted[:, i])), 0, i][0]
    y = data_is_subtracted[np.where((data_is_subtracted[:, i] >= 0) & (data[:, 0, i] <= vp[i]) & fromxongreaterthanzero(data_is_subtracted[:, i])), i][0]

    # fit the data
    res = optimize.leastsq(errfunc, np.asarray(p, dtype=np.float64), args=(x, y), full_output=True)
    (p1, pcov, infodict, errmsg, ier) = res

    # save electron temperatures
    tcold = p1[1]
    thot = p1[3]
    ncold = p1[0]/(elementary_charge*probe_area*np.sqrt(Boltzmann*tcold/(2*pi*electron_mass)))
    nhot = p1[2]/(elementary_charge*probe_area*np.sqrt(Boltzmann*thot/(2*pi*electron_mass)))

    temperatures[i, :] = [thot, tcold, nhot, ncold]

    # plot a dataset
    if i == datasettoplot-1:
        # plot the polynomial approximation to the second derivative between the maxima
        vpplot, = plt.plot(vpfitx, vpfit(vpfitx))
        # plot the linear ion saturation current fit
        ionsatplot, = plt.plot(ionsatx, ionsat[i](ionsatx))
        # plot the data
        dataplot, = plt.plot(data[:, 0, i], data[:, 1, i])
        # plot the ion saturation current corrected data
        data_is_subtractedplot, = plt.plot(data[:, 0, i], data_is_subtracted[:, i])
        # plot second derivative
        zaplot, = plt.plot(data[:, 0, i], za)
        # plot points used for electron temperature fitting
        plt.plot(x,y,'kx')
        plt.legend([dataplot, zaplot, vpplot, ionsatplot, data_is_subtractedplot], ['Data points', 'Second derivative', 'Approx. to second deriv.', 'Ion saturation current fit', 'Data points corr. by ion sat.'])

print(vp)
print(temperatures)
plt.show()
