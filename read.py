# This script will apply a SPIRAL Tap de-noising algorithm to CPL L0 raw photon counts
# in order to estimate the full signal from the photons detected by the instrument

from Denoise import descend
from read_routines import read_in_cats_l0_data
from matplotlib import pyplot as plt
from lidar import get_a_color_map
import pdb
import numpy as np
import h5py
from datetime import datetime
import matplotlib as mpl

vmin = -2
vmax = 100
now = datetime.now()

cls_path = "CATS_ISS_L0_D-M7.2-2017-10-25T18-20-45-T19-08-50.dat"
# noinspection SpellCheckingInspection
nbins = 480

data = read_in_cats_l0_data(cls_path, 12, nbins)['chan'][:, -1, :].T
print("Data shape = {}".format(data.shape))

#bg_rad = np.mean(data[384:450, :], axis=0)
#bg_rad = np.round(np.tile(bg_rad, (nbins, 1)))
#data = data - bg_rad


t_data = data[275:375, 15500:15600]

filt_data = descend(t_data, thresh=1e-15, iota=0.05, k_max=10)
"""
Plotting
"""


plt.subplot(2, 1, 1)
plt.imshow(t_data, cmap=get_a_color_map(), vmin=np.min(t_data), vmax=np.max(t_data), aspect='auto')

plt.colorbar()
plt.title("Before")

plt.subplot(2, 1, 2)

plt.imshow(filt_data, cmap=get_a_color_map(), vmin=np.min(filt_data), vmax=np.max(filt_data),
           aspect='auto')

plt.title("After")

plt.colorbar()


plt.savefig("filtered.png")
plt.show()




delta = datetime.now() - now
print("Script took {}".format(delta))


