import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.constants import Boltzmann, elementary_charge, pi, electron_mass, physical_constants
from scipy import optimize
import argparse
import os

# factor between eV and kelvin
conv_fac_K_eV = physical_constants['electron volt-kelvin relationship'][0]

# some general options

# percentage of minimum electron temperature difference to consider them as two populations
hot_cold_diff = 0.1
# abundance factor between the two population above which they should be combined.
# higher means only one resulting population less often
hot_cold_population_diff = 1e9
# minimum temperature in eV below which a population should be neglected and the two populations combined.
min_temp = 0.029 # room temperature
min_temp *= conv_fac_K_eV

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

parser.add_argument("--probe_diameter",
                    "-pd",
                    type=float,
                    help="Specify the diameter of the probe. Default is set to 0.0004 m",
                    default=0.0004)

parser.add_argument("--probe_length",
                    "-pl",
                    type=float,
                    help="Specify the length of the probe. Default is set to 0.003 m",
                    default=0.003)
parser.add_argument("--upper_limit_vp_fit",
                    "-ul",
                    type=float,
                    help="Specify the upper limit for the fitting routine. Default is set to None",
                    default=None)

# parse it
args = parser.parse_args()

if args.plot is not None:
    angle_to_plot = int(args.plot)
else:
    angle_to_plot = None
filename = args.filename
output = filename.split('.')[0] + '_results.txt'
probe_diam = args.probe_diameter  # probe diameter in m
probe_length = args.probe_length  # probe length in m
vp_upper_limit = args.upper_limit_vp_fit # define the upper limit for the fitting routine in Volt

# area of the langmuir probe
probe_area = probe_diam * pi * probe_length + probe_diam ** 2 * pi / 4  # in m^2
print('Langmuir probe geometry used for data evaluation:')
print('probe diameter = {}'.format(probe_diam))
print('probe length = {}'.format(probe_length))
print('probe area = {}'.format(probe_area))

# upper voltage limit for the fitting routine
print('upper limit for fitting routine in Volt = {}'.format(vp_upper_limit))

"""
Starting from here we set up some helper functions
"""


def from_x_on_greater_than_zero(array):
    """
    This function returns an array of booleans that indicate, whether all following data points are above zero. Example:
    [0 -1 -2 -3 -2 -1 0 1 2 1 0 -1 0 1 2 3 4 5]
    returns
    [F F F F F F F F F F F T T T T T T]
    :param array:
    :return: array with booleans
    """
    returnarray = [False] * array.shape[0]
    for j in np.arange(0, array.shape[0]):
        returnarray[j] = np.greater(array[j:], 0).all()
    return returnarray


# lambda to convert float with , as decimal point to real point
comma_to_dot = lambda s: float(s.decode("utf-8").replace(',', '.'))

# lambda that corresponds to complete current function as in http://dx.doi.org/10.1116/1.577344 eq. 1
ic_func = lambda p, x: p[0] * np.exp(elementary_charge * (x - vp[i]) / (Boltzmann * p[1])) + p[2] * np.exp(
    elementary_charge * (x - vp[i]) / (Boltzmann * p[3]))

# in the case of just one electron temperature we fit the same function, but with only one exponential func.
ic_func_single_temp = lambda p, x: p[0] * np.exp(elementary_charge * (x - vp[i]) / (Boltzmann * p[1]))

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

# check whether the file exists. if not, try to use it as a relative path.
if not os.path.exists(filename):
    filename = os.path.normcase(os.path.join(os.path.dirname(__file__), filename))

tempdata = np.loadtxt(filename, usecols=(0, 3, 4, 5), skiprows=7,
                      converters={0: comma_to_dot, 3: comma_to_dot, 4: comma_to_dot, 5: comma_to_dot}, dtype=np.float64)

# number of measurements = highest value of first column
nom = int(np.amax(tempdata, axis=0)[0])

# check whether all measurements have the same number of data points
if tempdata.shape[0] % nom != 0:
    print('Not an equal amount of data points for all measurements')
    quit(-1)

lines = int(tempdata.shape[0] / nom)

# create new array
data = np.zeros((lines - 1, 3, nom), dtype=np.float64)
tempdata2 = np.zeros((lines, 3, nom), dtype=np.float64)
for i in np.arange(0, nom):
    # copy from temporary data, but delete first column
    tempdata2[:, :, i] = np.delete(tempdata[np.where(tempdata[:, 0] == i + 1)], 0, 1)

    # also delete first data point, because it is an artifact
    data[:, :, i] = np.delete(tempdata2[:, :, i], 0, axis=0)

# sort array, in case the measurement software didn't save the points in order
for i in np.arange(0, nom):
    data[:, :, i] = data[data[:,0,i].argsort(), :, i]

# absolute error values
data[:, 2, :] = data[:, 1, :] * data[:, 2, :] / 100

# set tempdata to None so the GC can free memory
tempdata = None
tempdata2 = None

# prellocate some arrays / lists for data to be calculated

# preallocate array for the plasma potentials
vp = np.zeros((nom, 1), dtype=np.float64)
# preallocate array for the fit starting points
fit_start = np.zeros((nom, 1), dtype=np.float64)
# preallocate array for the floating potential
vf = np.zeros((nom, 1), dtype=np.float64)
# preallocate array for the EEPF
eepf = [None] * nom
# list for 1st order polynomial objects for ion saturation current
ionsat = [None] * nom
# list for errors of slope of ion saturation current
ionsat_slope_error = [None] * nom
# list for errors of intercept of ion saturation current
ionsat_intercept_error = [None] * nom
# numpy array for ion saturation current subtracted data
data_is_subtracted = np.zeros((lines - 1, nom), dtype=np.float64)
# preallocate array for electron temperatures (1 T_hot, 2 T_cold, 3 + 4 electron densities
temperatures = np.zeros((nom, 4), dtype=np.float64)
# preallocate array for the ion density
ion_density = np.zeros((nom, 1), dtype=np.float64)
# preallocate array for effective electron density
electron_density = np.zeros((nom, 1), dtype=np.float64)
# preallocate array for effective temperature
t_eff = np.zeros((nom, 1), dtype=np.float64)

# go through all measurements
for i in np.arange(0, nom):
    # first smooth the data and take the first derivative
    ea = np.gradient(savgol_filter(data[:, 1, i], 11, 2))
    # second derivative and smooth this again
    za = savgol_filter(np.gradient(ea), 21, 2)

    # this typically has a maximum (can be manually adjusted with vp_upper_limit), followed by a zero-crossing and a minimum
    # we search for the zero-crossing between the global maximum and minimum of the data, but not within 10% of the
    # border!

    tenpercent = int(np.round(lines / 10))

    if vp_upper_limit is None:
        maximum = data[np.where(za == np.max(za[tenpercent:-tenpercent])), 0, i]
        minimum = data[np.where(za == np.min(za[tenpercent:-tenpercent])), 0, i]
    else:
        vp_upper_limit_index = np.argmax(data[:, 0, i] > vp_upper_limit)
        maximum = data[np.where(za == np.max(za[tenpercent:vp_upper_limit_index])), 0, i]
        minimum = data[np.where(za == np.min(za[tenpercent:vp_upper_limit_index])), 0, i]

    # get the data points between max and min
    fitrangex = data[np.where((data[:, 0, i] >= maximum) & (data[:, 0, i] <= minimum)), 0, i][1]
    fitrangey = za[np.where((data[:, 0, i] >= maximum) & (data[:, 0, i] <= minimum))[1]]

    # sometimes we hit the wrong interval. don't throw errors in that case and mark the measurement as untrusted
    try:
        # fit the range between max and min with a third order polynomial
        vpfit = np.poly1d(np.polyfit(fitrangex, fitrangey, 3))
        vpfitx = np.linspace(maximum[0], minimum[0], 100)
    except Exception as e:
        print("Fehler in Angle {}:".format(i + 1))
        print(e)
        print('Minimum of second derivative: {}'.format(minimum))
        print('Maximum of second derivative: {}'.format(maximum))
        continue  # we can't use this angle

    # we take the zero crossing between max and min.
    for zerocrossing in vpfit.r:
        if (zerocrossing > maximum) & (zerocrossing < minimum):
            vp[i] = zerocrossing

    # next step: subtract ion saturation current
    # fit a linear function between -40 and 5 volts left of the zero crossing
    zerocrossingindex = np.where(np.diff(np.sign(data[:, 1, i])))[0]
    zerocrossingx = data[zerocrossingindex, 0, i] - 5

    # get the data points between max and min
    try:
        fitrangex = data[np.where((data[:, 0, i] >= -40) & (data[:, 0, i] <= zerocrossingx)), 0, i][0]
        fitrangey = data[np.where((data[:, 0, i] >= -40) & (data[:, 0, i] <= zerocrossingx)), 1, i][0]

        # calculate weights: 1 / error^2
        fiterrors = data[np.where((data[:, 0, i] >= -40) & (data[:, 0, i] <= zerocrossingx)), 2, i][0]
        fitweights = 1/fiterrors**2

    except ValueError:
        print('Fucked up measurement at angle {}'.format(i + 1))
        quit(-1)
    if fitrangex.shape[0] == 0:
        print('No data points for linear fit between -40 and -10 V found for angle {}'.format(i + 1))
        continue  # we cannot use this angle

    linfit, linfitcov = np.polyfit(fitrangex, fitrangey, 1, w = fitweights, cov = True)
    ionsat[i] = np.poly1d(linfit)
    ionsat_slope_error[i], ionsat_intercept_error[i] = np.diagonal(linfitcov)
    ionsatx = np.linspace(-40, zerocrossingx, 100)

    # create a new dataset, where we subtract the ion saturation current
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

    bounds_lower = [0] * 4
    bounds_upper = [100, 1e8, 100, 1e8]

    # select the data points to be fitted. three conditions:
    # 1) ion saturation corrected current has to be above zero
    # 2) only data points below the plasma potential
    # 3) using the above function from_x_on_greater_than_zero we avoid data points above zero which happened due to the
    # correction

    x = data[np.where((data_is_subtracted[:, i] >= 0) & (data[:, 0, i] <= vp[i]) & from_x_on_greater_than_zero(
        data_is_subtracted[:, i])), 0, i][0]
    y = data_is_subtracted[
        np.where((data_is_subtracted[:, i] >= 0) & (data[:, 0, i] <= vp[i]) & from_x_on_greater_than_zero(
            data_is_subtracted[:, i])), i][0]
    yerr = data[np.where((data_is_subtracted[:, i] >= 0) & (data[:, 0, i] <= vp[i]) & from_x_on_greater_than_zero(
        data_is_subtracted[:, i])), 2, i].T.flatten()

    # we interpolate linearly between the first point used for the fits (x[0]) and the one before
    index_before_fit_start = np.where(data[:, 0, i] == x[0])[0][0] - 1
    x_before_fit_start = data[index_before_fit_start, 0, i]
    y_before_fit_start = data_is_subtracted[index_before_fit_start, i]
    fit_start[i] = x_before_fit_start - (x[0] - x_before_fit_start) / (y[0] - y_before_fit_start) * y_before_fit_start

    # similar procedure to calculate the floating potential:
    # 1) probe current has to be above zero (the uncorrected one, as opposed to above)
    # --> zero crossing of probe current
    vf_at_datapoint = data[np.where(from_x_on_greater_than_zero(data[:, 1, i])), 0, i][0][0]
    index_before_vf = np.where(data[:, 0, i] == vf_at_datapoint)[0][0] - 1
    x_before_vf = data[index_before_vf, 0, i]
    y_before_vf = data[index_before_vf, 1, i]
    vf[i] = x_before_vf - (vf_at_datapoint - x_before_vf) / (data[np.where(data[:, 0, i] == vf_at_datapoint), 1, i][0][0] - y_before_vf) * y_before_vf

    # fit the data
    fit_success = False
    try:
        res = optimize.least_squares(errfunc, np.asarray(p, dtype=np.float64), args=(x, y), bounds=(bounds_lower, bounds_upper))
        p1 = res.x
        fit_success = True
    except TypeError:
        print('Warning: not enough fitting points between floating and plasma potential for angle {}'.format(i + 1))
        p1 = [None] * 4

    if fit_success:
        # inspect and save electron temperatures
        # if we happen to have only one electron temperature in the measurement, the fit converges to almost the same
        # electron temperature for t_hot and t_cold. hence we check, whether they are within a certain percentage of one
        # another and if so, we add the currents (as they are the same and save it as [t|n]_cold
        # factor 1000000000 because of mA current signal and m^-3 to cm^-3 
        t_cold = p1[1]
        t_hot = p1[3]
        n_cold = p1[0] / (elementary_charge * probe_area) * np.sqrt(
            2 * pi * electron_mass / (Boltzmann * t_cold)) / 1e9
        n_hot = p1[2] / (elementary_charge * probe_area) * np.sqrt(
            2 * pi * electron_mass / (Boltzmann * t_hot)) / 1e9

        # check whether they switched places
        if t_cold > t_hot:
            t_cold, t_hot = t_hot, t_cold
            n_cold, n_hot = n_hot, n_cold

        # calculate total electron density 
        electron_density[i] = n_cold + n_hot

        # check whether the two electron temperature are
        # 1) within a small window or
        # 2) whether one population is much more abundant then the other. or
        # 3) either of the temperatures is very small
        # if any of that is the case, we fit again with only one population.
        if ((t_hot / t_cold < (1 + hot_cold_diff)) & (t_hot / t_cold > (1 - hot_cold_diff))) | ((n_cold/n_hot > hot_cold_population_diff) | (n_hot/n_cold > hot_cold_population_diff)) | (t_cold < min_temp):
            print("Only one popuplation in angle {}:".format(i + 1))
            print('Old:\nT_cold: {} N_cold {}\nT_hot: {} N_hot {}'.format(t_cold, n_cold, t_hot, n_hot))

            # new guessed parameters
            p = [0] * 2
            if t_cold < min_temp:
                p[0] = n_hot * 1e9 / np.sqrt(
                    2 * pi * electron_mass / (Boltzmann * t_hot)) * (
                       elementary_charge * probe_area)  # fit should be around the sum of the two
                p[1] = t_hot
            else:
                p[0] = (n_hot + n_cold) * 1e9 / np.sqrt(2 * pi * electron_mass / (Boltzmann * (n_hot*t_hot + n_cold*t_cold)/(n_hot+n_cold))) * (elementary_charge * probe_area)# fit should be around the sum of the two
                p[1] = (n_hot*t_hot + n_cold*t_cold)/(n_hot+n_cold)

            bounds_lower = [0] * 2
            bounds_upper = [200, 1e8]

            # new function with just one exponential
            errfunc = lambda p, x, y: ic_func_single_temp(p, x) - y
            res = optimize.least_squares(errfunc, np.asarray(p, dtype=np.float64), args=(x, y), bounds=(bounds_lower, bounds_upper))
            p1 = res.x

            n_new = p1[0] / (elementary_charge * probe_area) * np.sqrt(
                2 * pi * electron_mass / (Boltzmann * t_cold)) / 1000000000
            t_new = p1[1]

            print('New:\nT_cold: {} N_cold {}'.format(t_new, n_new))

            # "calculate" total electron density
            electron_density[i] = n_new

            # in order to decide which population (hot, cold) the new fit belongs to, we calculate the median of the
            # temperatures in both populations so far and simply compare the distance to either.
            # this has the disadvantage of misassignments towards low angles when the statistics is not good yet.
            # however, yields better results than always assigning to either hot or cold.

            # uncomment to see population statistics so far
            #print('Cold: {} +- {}'.format(np.nanmedian(temperatures[0:i-1,1]), np.nanstd(temperatures[0:i-1,1])))
            #print('Hot: {} +- {}'.format(np.nanmedian(temperatures[0:i - 1, 0]), np.nanstd(temperatures[0:i - 1, 0])))
            if np.abs(np.nanmedian(temperatures[0:i-1,1])-t_new / conv_fac_K_eV) > np.abs(np.nanmedian(temperatures[0:i-1,0])-t_new / conv_fac_K_eV):
                # closer to hot
                t_cold = None
                n_cold = None
                t_hot = t_new
                n_hot = n_new
            else:
                # closer to cold
                t_hot = None
                n_hot = None
                t_cold = t_new
                n_cold = n_new

        temperatures[i, :] = [t_hot, t_cold, n_hot, n_cold]

        # calculate effective temperature according to http://dx.doi.org/10.1119/1.2772282
        if t_hot is not None and t_cold is None:
            t_eff[i] = t_hot
        elif t_hot is None and t_cold is not None:
            t_eff[i] = t_cold
        else:
            t_eff[i] = ((n_cold / (n_cold + n_hot)) * (1 / t_cold) + (n_hot / (n_cold + n_hot)) * (1 / t_hot)) ** (-1)

        # calculate ion density according to http://dx.doi.org/10.1116/1.1515800
        # factor 1000000000 because of mA current signal and m^-3 to cm^-3 
        mass_argon = 39.96238 * physical_constants['atomic mass constant'][0]
        ion_density[i] = np.abs(ionsat[i](vp[i])) / (0.6 * elementary_charge ** (3 / 2) * probe_area) * np.sqrt(
            elementary_charge * mass_argon / (Boltzmann * t_eff[i])) / 1000000000

    # calculate electron energy probability function according to http://dx.doi.org/10.1063/1.4905901
    # eepf is just a python list, because we the number of datapoints for each angle can be different. each entry of
    # eepf is the a numpy array with shape (data points, 2)
    eepf_array = np.zeros((data[np.where((data[:, 0, i] < vp[i]) & (data[:, 0, i] > fit_start[i])), 0, i].shape[1], 2),
                          dtype=np.float64)
    eepf_array[:, 0] = vp[i] - data[np.where((data[:, 0, i] < vp[i]) & (data[:, 0, i] > fit_start[i])), 0, i]
    eepf_array[:, 1] = za[np.where((data[:, 0, i] < vp[i]) & (data[:, 0, i] > fit_start[i]))] * (
        2 * electron_mass / (elementary_charge ** 2 * probe_area)) * np.sqrt(2 * elementary_charge / electron_mass)
    eepf[i] = eepf_array

    if fit_success:
        # convert temperatures to eV
        temperatures[i, 0] = temperatures[i, 0] / conv_fac_K_eV
        temperatures[i, 1] = temperatures[i, 1] / conv_fac_K_eV

    # plot a dataset
    if angle_to_plot is not None:
        if angle_to_plot - 1 == i:
            fig1 = plt.figure()
            ax0 = fig1.add_subplot(2, 1, 1)
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
            fitpointsplot = ax0.errorbar(x, y, yerr=yerr, fmt='x', ecolor='g', capthick=1)
            ax0.legend([dataplot, zaplot, vpplot, ionsatplot, data_is_subtractedplot, fitpointsplot],
                       ['Data points', 'Second derivative', 'Approx. to second deriv.', 'Ion saturation current fit',
                        'Data points corr. by ion sat.', 'Data points used for fitting'])
            ax0.set_title('I-V characteristic for angle {}'.format(angle_to_plot))
            ax0.set_xlabel('Probe Voltage (V)')
            ax0.set_ylabel('Current (mA)')

            # subplot for EEPF
            ax01 = fig1.add_subplot(2, 1, 2)
            eepf_plot, = ax01.semilogy(eepf[i][:, 0], eepf[i][:, 1])
            ax01.set_title('EEPF for angle {}'.format(angle_to_plot))
            ax01.set_ylabel('A.U.')
            ax01.set_xlabel('Energy (eV)')
            fig1.tight_layout()
            fig1.suptitle('Filename {}'.format(filename))

# export all data to a file
export_values = np.concatenate(
    (np.arange(1, nom + 1).reshape((nom, 1)), vp, temperatures, ion_density, fit_start, electron_density, vf), axis=1)
np.savetxt(output,
           export_values,
           fmt=('%d', '%10.6f', '%10.2f', '%10.2f', '%1.4e', '%1.4e', '%1.4e', '%10.6f', '%1.4e', '%10.6f'),
           delimiter='\t',
           header='Measurement\tPlasma Potential\tT_hot\tT_cold\tn_hot\tn_cold\tn_ion\tFit starting point\tEffective Electron Density\tFloating Potential')

# make a nice overview plot using the export_values array
fig2 = plt.figure()
ax1 = fig2.add_subplot(231)
ax1.plot(export_values[:, 0], export_values[:, 1], 'x')
ax1.set_title('Plasma Potential')
ax1.set_xlabel('Angle')
ax1.set_ylabel(r'$V_p$ (V)')

ax2 = fig2.add_subplot(235)
thotplot, = ax2.plot(export_values[:, 0], export_values[:, 2], 'x')
tcoldplot, = ax2.plot(export_values[:, 0], export_values[:, 3], 'x')
ax2.set_title('Electron Temperature')
ax2.set_xlabel('Angle')
ax2.set_ylabel(r'$T_e (eV)$')
ax2.set_ylim([0, 15])
ax2.legend([thotplot, tcoldplot], ['Hot Electrons', 'Cold Electrons'])

ax3 = fig2.add_subplot(233)
n_ion_plot, = ax3.plot(export_values[:, 0], export_values[:, 6], 'x')
ax3.set_title('Ion Density')
ax3.set_xlabel('Angle')
ax3.set_ylabel(r'$n_{ion} (cm^{-3})$')

ax4 = fig2.add_subplot(232)
nhotplot, = ax4.plot(export_values[:, 0], export_values[:, 4], 'x')
ncoldplot, = ax4.plot(export_values[:, 0], export_values[:, 5], 'x')
ax4.set_title('Electron Density')
ax4.set_xlabel('Angle')
ax4.set_ylabel(r'$n_e (cm^{-3})$')
ax4.legend([nhotplot, ncoldplot], ['Hot Electrons', 'Cold Electrons'])

ax5 = fig2.add_subplot(234)
ax5.plot(export_values[:, 0], export_values[:, 9], 'x')
ax5.set_title('Floating Potential')
ax5.set_xlabel('Angle')
ax5.set_ylabel(r'$V_f$ (V)')

ax6 = fig2.add_subplot(236)
ax6.plot(export_values[:, 0], export_values[:, 8], 'x')
ax6.set_title('Effective Electron Density')
ax6.set_xlabel('Angle')
ax6.set_ylabel(r'$n_{e,eff} (cm^{-3})$')

fig2.tight_layout()
fig2.suptitle('Filename {}'.format(filename))
plt.show()
