# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 14:48:12 2016

@author: ivan
"""

import numpy as np
from  scipy.signal import argrelmax
import matplotlib.pyplot as plt

Np = 400 # number of pyramide neurons
Nb = 20   # number of basket cells
Nolm = 20 # number of olm cells
Ns = 400 # number synapses between pyramide cells
NSG = 20 # number of spike generators
Ns2in = 400 # 


path = "/home/ivan/Data/modeling_septo_hippocampal_model/only_one_rhytm2/"

number = "21"
file = path + number + "_lfp.npy"
fd = 500

lfp = np.load(file)
V = np.load(path +  number + "_V.npy")
firing = np.load(path +  number + "_firing.npy")
lfp = -lfp[0:-1:20]
t = np.linspace(0, lfp.size/fd, lfp.size)
lfp_fft = 2 * np.fft.rfft(lfp) / lfp.size

w = np.fft.rfftfreq(lfp.size, 1/fd )
Z = np.exp(-0.05*w) + 0.002

lfp_fft *= Z
#lfp_fft[(w>11)&(w<13)] = 0
lfp_filtred = np.fft.irfft(lfp_fft)
idx_max = argrelmax(lfp_filtred, order=20)[0]
idx_min = argrelmax(-lfp_filtred, order=20)[0]

"""
Z = np.zeros(w.size) #
Z[(w >= 4)&(w <= 12)] = 1
lfp_fft2 = lfp_fft * Z
lfp_teta = np.fft.irfft(lfp_fft2)
lfp_phase = np.angle( hilbert(lfp_teta) )

idx_min = idx_min[ np.abs(lfp_phase[idx_min]) > 2.5 ]
idx_max = idx_max[ np.abs(lfp_phase[idx_max]) < 0.8 ]
"""

assymetry_index = np.array([], dtype=float)
n = np.min([idx_max.size, idx_min.size])
for idx in range(n):
    if (idx == 0 or idx+1 == n):
        continue

    if (idx_max[idx] < idx_min[idx] and idx_max[idx+1] > idx_min[idx]):
        asind = np.log( (idx_max[idx] - idx_min[idx-1]) / (idx_min[idx] - idx_max[idx]) ) 
        assymetry_index = np.append(assymetry_index, asind)

    if (idx_max[idx] > idx_min[idx] and idx_min[idx+1] > idx_max[idx]):
        asind = np.log( (idx_max[idx] - idx_min[idx]) / (idx_min[idx+1] - idx_max[idx]) )
        assymetry_index = np.append(assymetry_index, asind)

plt.hist(assymetry_index, 10, normed=1, facecolor='blue', alpha=0.75)
plt.xlabel('Asymmetry index')
plt.ylabel('Theta cycles')
# plt.title('')
v = V[51]["soma"][0:-1:20]
plt.figure()
plt.plot(t, lfp)
#plt.xlim(0., 1.)
plt.figure()
plt.subplot(111)
plt.plot(t, lfp_filtred, color="black", linewidth=10)

"""
for i in idx_max:
    plt.plot([t[i], t[i]], [-0.2, 0.2], color="blue", linestyle="dashed", linewidth=2)

for i in idx_min:
    plt.plot([t[i], t[i]], [-0.2, 0.2], color="blue", linestyle="dashed", linewidth=2)
"""
"""
plt.scatter(t[idx_max], lfp_filtred[idx_max], color="red")
plt.scatter(t[idx_min], lfp_filtred[idx_min], color="green")
"""

#plt.xlim(0., 1.)
#plt.subplot(212)
#plt.plot(t, v, "r")
#plt.subplot(313)

plt.figure()
firing[0, :] *= 0.001 
cum_it = Np
sl = firing[1, :] <= cum_it
pyr_line, = plt.plot(firing[0, sl], firing[1, sl], '.b', label='Pyramide')
        
sl = (firing[1, :] > cum_it) & (firing[1, :] <= cum_it + Nb)
    
basket_line, = plt.plot(firing[0, sl], firing[1, sl], '.g', label='Basket')
cum_it += Nb
sl = (firing[1, :] > cum_it) & (firing[1, :] <= cum_it + Nolm)
        
olm_line, = plt.plot(firing[0, sl], firing[1, sl], '.m', label='OLM')
cum_it += Nolm
sl = (firing[1, :] > cum_it)
        
septal_line, = plt.plot(firing[0, sl], firing[1, sl], '.r', label='Septum')
        
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


plt.figure()
plt.plot(w, np.abs(lfp_fft)/lfp_filtred.size)
plt.xlim(1, 50)
plt.ylim(0, 0.05)
