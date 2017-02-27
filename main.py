import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.constants import Boltzmann, elementary_charge, pi, electron_mass
from scipy import optimize
import argparse

# some general options

# area of the langmuir probe
probe_area = 0.000001  # 1mm^2
# percentage of minimum electron temperature difference to consider them as two populations
hot_cold_diff = 0.05

# using the argparse module to make use of command line options
parser = argparse.ArgumentParser(description="Evaluation script for langmuir measurements")

# add commandline options
parser.add_argument("--filename",
                    "-f",
                    help="Specify a filename to be evaluated. Defaults to testdata.txt",
                    default='testdata.txt')
parser.add_argument("--plot",
                    "-p",
                    help="Specify an angle to be plotted in detail",
                    default=None,
                    type=int)
parser.add_argument("--output",
                    "-o",
                    help="Specify a filename for the output data. Defaults to results.txt",
                    default='results.txt')

# parse it
args = parser.parse_args()

angle_to_plot = int(args.plot)
filename = args.filename
output = args.output

"""
Starting from here we set up some helper functions
"""


def from_x_on_greater_than_zero(array):
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
comma_to_dot = lambda s: float(s.decode("utf-8").replace(',', '.'))

# lambda that corresponds to complete current function as in http://dx.doi.org/10.1116/1.577344 eq. 1
ic_func = lambda p, x: p[0] * np.exp(elementary_charge * (x - vp[i]) / (Boltzmann * p[1])) + p[2] * np.exp(
    elementary_charge * (x - vp[i]) / (Boltzmann * p[3]))

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
tempdata = np.loadtxt(filename, usecols=(0, 3, 4, 5), skiprows=7,
                      converters={0: comma_to_dot, 3: comma_to_dot, 4: comma_to_dot, 5: comma_to_dot}, dtype=np.float64)

# number of measurements = highest value of first column
nom = int(np.amax(tempdata, axis=0)[0])

# check whether all measurements have the same number of data points
if tempdata.shape[0] % nom != 0:
    quit('Not an equal amount of data points for all measurements')

lines = int(tempdata.shape[0] / nom)

# create new array
data = np.zeros((lines - 1, 3, nom), dtype=np.float64)
tempdata2 = np.zeros((lines, 3, nom), dtype=np.float64)
for i in np.arange(0, nom):
    # copy from temporary data, but delete first column
    tempdata2[:, :, i] = np.delete(tempdata[np.where(tempdata[:, 0] == i + 1)], 0, 1)

    # also delete first data point, because it is an artifact
    data[:, :, i] = np.delete(tempdata2[:, :, i], 0, axis=0)

# set tempdata to None so the GC can free memory
tempdata = None
tempdata2 = None

# prellocate some arrays / lists for data to be calculated

# preallocate array for the plasma potentials
vp = np.zeros((nom, 1), dtype=np.float64)
# list for 1st order polynomial objects for ion saturation current
ionsat = [None] * nom
# numpy array for ion saturation current subtracted data
data_is_subtracted = np.zeros((lines - 1, nom), dtype=np.float64)
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
    tenpercent = int(np.round(lines / 10))

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
        return data[np.where(data[:, 0, i] == a), 1, i] - ionsat[i](a)


    data_is_subtracted[:, i] = np.apply_along_axis(subtract_ion_sat_current, 0, data[:, 0, i])

    # starting from here we fit electron temperatures and currents
    errfunc = lambda p, x, y: ic_func(p, x) - y

    # starting values
    p = [0] * 4
    p[0] = 1
    p[1] = 29000  # 2.5 eV in K
    p[2] = 1
    p[3] = 145100  # 12.5 eV in K

    # select the data points to be fitted. three conditions:
    # 1) ion saturation corrected current has to be above zero (above floating potential)
    # 2) only datapoints below the plasma potential
    # 3) using the above function from_x_on_greater_than_zero we avoid datapoints above zero which happened due to the
    # correction

    x = data[np.where((data_is_subtracted[:, i] >= 0) & (data[:, 0, i] <= vp[i]) & from_x_on_greater_than_zero(
        data_is_subtracted[:, i])), 0, i][0]
    y = data_is_subtracted[np.where((data_is_subtracted[:, i] >= 0) & (data[:, 0, i] <= vp[i]) & from_x_on_greater_than_zero(
        data_is_subtracted[:, i])), i][0]

    # fit the data
    res = optimize.leastsq(errfunc, np.asarray(p, dtype=np.float64), args=(x, y), full_output=True)
    (p1, pcov, infodict, errmsg, ier) = res

    # inspect and save electron temperatures
    # if we happen to have only one electron temperature in the measurement, the fit converges to almost the same
    # electron temperature for thot and tcold. hence we check, whether they are within 2% of one another and if so,
    # we add the currents (as they are the same and save it as [t|n]cold
    tcold = p1[1]
    thot = p1[3]
    ncold = p1[0] / (elementary_charge * probe_area * np.sqrt(Boltzmann * tcold / (2 * pi * electron_mass)))
    nhot = p1[2] / (elementary_charge * probe_area * np.sqrt(Boltzmann * thot / (2 * pi * electron_mass)))

    # check whether they switches places
    if tcold > thot:
        tcold, thot = thot, tcold
        ncold, nhot = nhot, ncold

    if (thot / tcold < (1 + hot_cold_diff)) & (thot / tcold > (1 - hot_cold_diff)):
        thot = None
        ncold += nhot
        nhot = None

    temperatures[i, :] = [thot, tcold, nhot, ncold]

    # plot a dataset
    if angle_to_plot:
        if angle_to_plot - 1 == i:
            fig1 = plt.figure()
            ax0 = fig1.add_subplot(1, 1, 1)
            # plot the polynomial approximation to the second derivative between the maxima
            vpplot, = ax0.plot(vpfitx, vpfit(vpfitx))
            # plot the linear ion saturation current fit
            ionsatplot, = ax0.plot(ionsatx, ionsat[i](ionsatx))
            # plot the data
            dataplot, = ax0.plot(data[:, 0, i], data[:, 1, i])
            # plot the ion saturation current corrected data
            data_is_subtractedplot, = ax0.plot(data[:, 0, i], data_is_subtracted[:, i])
            # plot second derivative
            zaplot, = ax0.plot(data[:, 0, i], za)
            # plot points used for electron temperature fitting
            fitpointsplot, = ax0.plot(x, y, 'kx')
            ax0.legend([dataplot, zaplot, vpplot, ionsatplot, data_is_subtractedplot, fitpointsplot],
                       ['Data points', 'Second derivative', 'Approx. to second deriv.', 'Ion saturation current fit',
                        'Data points corr. by ion sat.', 'Data points used for fitting'])
            ax0.set_title('I-V characteristic for angle {}'.format(angle_to_plot))
            ax0.set_xlabel('Probe Voltage (V)')
            ax0.set_ylabel('Current (mA)')
            fig1.tight_layout()

# export all data to a file
export_values = np.concatenate((np.arange(1, 42).reshape((41, 1)), vp, temperatures), axis=1)
np.savetxt(output,
           export_values,
           fmt=('%d', '%10.6f', '%10.2f', '%10.2f', '%1.4e', '%1.4e'),
           delimiter='\t',
           header='Measurement\tPlasma Potential\tT_hot\tT_cold\tn_hot\tn_cold')

# make a nice overview plot using the export_values array
fig2 = plt.figure()
ax1 = fig2.add_subplot(221)
ax1.plot(export_values[:, 0], export_values[:, 1], 'x')
ax1.set_title('Plasma Potential')
ax1.set_xlabel('Angle')
ax1.set_ylabel(r'$V_p$ (V)')

ax2 = fig2.add_subplot(222)
thotplot, = ax2.plot(export_values[:, 0], export_values[:, 2], 'x')
tcoldplot, = ax2.plot(export_values[:, 0], export_values[:, 3], 'x')
ax2.set_title('Electron Temperature')
ax2.set_xlabel('Angle')
ax2.set_ylabel(r'$T_e$ (K)')
ax2.set_ylim([0, 150000])
ax2.legend([thotplot, tcoldplot], ['Hot Electrons', 'Cold Electrons'])

ax3 = fig2.add_subplot(223)
nhotplot, = ax3.plot(export_values[:, 0], export_values[:, 4], 'x')
ncoldplot, = ax3.plot(export_values[:, 0], export_values[:, 5], 'x')
ax3.set_title('Electron Density')
ax3.set_xlabel('Angle')
ax3.set_ylabel(r'$n_e$ [nofuckinidea]')
ax3.legend([nhotplot, ncoldplot], ['Hot Electrons', 'Cold Electrons'])

fig2.tight_layout()
plt.show()
