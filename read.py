# This script will apply a SPIRAL Tap de-noising algorithm to CPL L0 raw photon counts
# in order to estimate the full signal from the photons detected by the instrument

import Denoiser
from read_routines import read_in_cats_l0_data
from matplotlib import pyplot as plt
from lidar import get_a_color_map
import pdb
import numpy as np
import h5py
from datetime import datetime
from Denoiser import descend
import matplotlib as mpl

vmin = -2
vmax = 100
now = datetime.now()
cls_path = "C:\\Users\\drusi\\OneDrive\\Desktop\\spiralTap\\data\\CATS_ISS_L0_D-M7.2-2017-10-25T18-20-45-T19-08-50.dat"


# noinspection SpellCheckingInspection
def spiral(cls_path):
    nbins = 480

    # Export sample photon count to file for later use
    # np.savetxt('data.csv', chan_532_photon_counts[2], delimiter=',')

    data = read_in_cats_l0_data(cls_path,12,nbins)['chan'][:,-1,:].T
    print("Data shape = {}".format(data.shape))

    bg_rad = np.mean(data[384:450,:], axis=0)
    bg_rad = np.round(np.tile(bg_rad,(nbins,1)))
    data = data - bg_rad
    data[data <= 0] = 0.0001

    tmpMat = np.zeros_like(data)
    rows, cols = 0, 0

    t_data = data[:,15000:15100]
  
    
    for prof in range(0, t_data.shape[1]):
        k_prof = t_data[:, prof]
        f0 = np.copy(k_prof) + np.random.randint(1,10,(k_prof.size))

        f0[f0 <= 0] = 1

        #in_data = np.ravel(t_data)
        #f0 = np.ravel(f0)

        filt_data= descend(data=k_prof, f0=f0, thresh=1e-5, iota=100000, k_max=20)
        tmpMat[:, prof] - filt_data

    return t_data, tmpMat


t_data, filt_data = spiral(cls_path)

"""
Plotting
"""
fig = plt.figure(figsize=(10,10))
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

mpl.rc('font', **font)

plt.subplot(2, 1, 1)
plt.imshow(t_data, cmap=get_a_color_map(), vmin=vmin, vmax=vmax, aspect='auto')

plt.colorbar()
plt.title("Before")

plt.subplot(2, 1, 2)

plt.imshow(filt_data, cmap=get_a_color_map(), vmin=vmin, vmax=vmax,
           aspect='auto')

plt.title("After")

plt.colorbar()


plt.savefig("filtered.png")

#profile to inspect for plot

xs1 = t_data[:, 1]

ys1 = range(0, data.shape[0])

xs2 = t_data[:, 1]
ys2 = range(0, data.shape[0])

fig = plt.figure(figsize=(10,30))

plt.plot(xs1, ys1, label="Before", linewidth=7.0)
plt.plot(xs2, ys2, label="After")
L = plt.legend()
plt.savefig("inspect.png")

delta = datetime.now() - now
print("Script took {}".format(delta))


