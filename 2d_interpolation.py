import numpy as np
import glob
import os
import argparse
import matplotlib.pyplot as plt
from scipy.interpolate import SmoothBivariateSpline, interp1d

# using the argparse module to make use of command line options
parser = argparse.ArgumentParser(description="2D plot script for langmuir measurements")

# add commandline options
parser.add_argument("--folder",
                    "-f",
                    help="Specify a folder of files to be evaluated")

# parse it
args = parser.parse_args()

folder = args.folder

def fill_nan(A):
    '''
    interpolate to fill nan values
    '''
    inds = np.arange(A.shape[0])
    good = np.where(np.isfinite(A))
    f = interp1d(inds[good], A[good], bounds_error=False)
    B = np.where(np.isfinite(A), A, f(inds))
    return B


# column to plot
plot_column = 0

# folder to read from
fnames = glob.glob(os.path.join(os.path.normpath(folder), '*_results.txt'))

# heights for files
heights = [None] * len(fnames)

# look for the largest dataset
x = 0
y = 0
for filename in fnames:
    temp_data = np.loadtxt(filename, skiprows=1, dtype=np.float64)
    print(temp_data.shape)
    if temp_data.shape[0] > y:
        y = temp_data.shape[0]
    if temp_data.shape[1] > x:
        x = temp_data.shape[1]

print('Array size x: {} y: {}'.format(x, y))

# create empty array
data = np.zeros((y, x, len(fnames)), dtype=np.float64)

# read files
i = 0
for filename in fnames:
    # check whether the file exists. if not, try to use it as a relative path.
    if not os.path.exists(filename):
        filename = os.path.normcase(os.path.join(os.path.dirname(__file__), folder, filename))

    # read heights from filename - currently windows specific
    measurement_name = filename.split('_')[0]
    if measurement_name[-2:-1] == '1':
        heights[i] = 10
    elif measurement_name[-2:-1] == '2':
        heights[i] = 15
    elif measurement_name[-2:-1] == '3':
        heights[i] = 20
    elif measurement_name[-2:-1] == '4':
        heights[i] = 30
    elif measurement_name[-2:-1] == '5':
        heights[i] = 40

    temp = np.loadtxt(filename, skiprows=1, dtype=np.float64)
    data[0:temp.shape[0], :, i] = temp
    # interpolate nans in the row
    data[:, plot_column, i] = fill_nan(data[:, plot_column, i])

    i += 1

# number of heights
noh = data.shape[2]

# number of angles
noa = data.shape[0]

# create new x and y axes
x = np.arange(0, noa)
y = np.arange(min(heights), max(heights) + 1, noa)
plot_array = np.zeros((max(x) + 1, max(y) + 1), dtype=np.float64)

xinput = np.tile(x, noh)
yinput = np.repeat(heights, noa)
zinput = data[:, plot_column, :].T.flatten()
# f = interp2d(x = xinput, y = yinput, z = zinput, kind='cubic')

# interpolate
f = SmoothBivariateSpline(x=xinput, y=yinput, z=zinput)

xnew = np.arange(min(x), max(x) + 0.5, 1)
ynew = np.arange(min(heights), max(heights), 1)

# znew = f(xnew, ynew)
znew = np.zeros((len(ynew), len(xnew)))

# evaluate interpolation function on new grid
j = 0
i = 0
for y in ynew:
    for x in xnew:
        znew[i, j] = f.ev(x, y)
        j += 1
    i += 1
    j = 0

plt.imshow(znew, interpolation='hamming', origin='lower')
#
plt.colorbar()
plt.title('Plasma Potential')
plt.xlabel('Angle')
plt.ylabel('Distance from target (mm)')
plt.show()
