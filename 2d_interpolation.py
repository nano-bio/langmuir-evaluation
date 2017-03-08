import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from scipy.interpolate import SmoothBivariateSpline, interp1d


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
plot_column = 5

# folder to read from
fnames = glob.glob('results/*')

# heights for files
heights = [None] * len(fnames)

# read first file to know dimensions
temp_data = np.loadtxt(fnames[0], skiprows=1, dtype=np.float64)

# create empty array
data = np.zeros((temp_data.shape[0], temp_data.shape[1], len(fnames)), dtype=np.float64)

# read files
i = 0
for filename in fnames:
    # check whether the file exists. if not, try to use it as a relative path.
    if not os.path.exists(filename):
        filename = os.path.normcase(os.path.join(os.path.dirname(__file__), filename))

    # read heights from filename - currently windows specific
    heights[i] = int(filename.split('.')[0].split('\\')[1])
    data[:, :, i] = np.loadtxt(filename, skiprows=1, dtype=np.float64)
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

plt.imshow(znew, interpolation='bicubic', origin='lower')
#
plt.colorbar()
plt.title('Plasma Potential')
plt.xlabel('Angle')
plt.ylabel('Distance from target (mm)')
plt.show()
