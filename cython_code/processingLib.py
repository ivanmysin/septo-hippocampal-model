# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 14:10:11 2014
library for data proceccing 
@author: ivan
"""
import numpy as np
from scipy.signal import argrelmax
import scipy.stats as stat
import scipy.signal as sig 

def mycwt (signal, scales):
    # print "hello"
    # compute complex morlet transform with fft algorhythm
    # function return vector of complex numbers 
    pi = np.pi
    N = signal.size
    J = scales.size
    signal_fft = np.fft.fft(signal)

    # pre-compute the Fourier transform of the Morlet wavelet 
    w0 = 5 # central frequency of morlet wavelet
    w = np.array( range(N//2), dtype=np.float32 ) / N # vector of frequencies Fourier of morlet function
    # compute the CWT
    morletft = np.zeros(N, dtype = np.float32)
    X = np.zeros ((J, N), dtype=np.complex64) #np.complex128
    i = 0
    while (i < J):
       morletft[:N//2] = np.sqrt(2 * pi * scales[i]) * np.exp(-(scales[i] * 2 * pi * w - w0)**2 / 2.0) 
       tmp = np.fft.ifft(signal_fft * morletft)
       X[i, :] = tmp.copy()
       X[i, :] = X[i, :] / np.sqrt(scales[i])
       i = i + 1
    #   print i, '/', J
    return X
#####################################################################   
def computemycwt (fd, signal):
    """
compute parameters for function mycwt and run it
fd is frequency of discretization
signal is sequence of samples
function return wavelet coefficients for morlet transform 
    """
    # N = signal.size
    fr = np.arange(1, 80)         # np.linspace(1, 200, 400); # vector of frequencies
    w0 = 5    
    scales = fd*w0/fr/2/np.pi

    # J = 200    
    # scales = np.asarray([2**(i * 0.1) for i in range(J)])

    coef = mycwt(signal, scales)
    #t = np.arange (0, N, 1/fd, dtype = np.float64)
    return fr, coef
######################################################################
# calculation of mututial information

def calc_MI(X,Y,bins):

   c_XY = np.histogram2d(X,Y,bins)[0]
   c_X = np.histogram(X,bins)[0]
   c_Y = np.histogram(Y,bins)[0]

   H_X = shan_entropy(c_X)
   H_Y = shan_entropy(c_Y)
   H_XY = shan_entropy(c_XY)

   MI = H_X + H_Y - H_XY
   return MI

def shan_entropy(c):
    c_normalized = c/float(np.sum(c))
    c_normalized = c_normalized[np.nonzero(c_normalized)]
    H = -sum(c_normalized* np.log2(c_normalized))  
    return H
####################################################################    
def correlationfunction (x, y, shifts):
    """
    function return correlation between x and y at different shifts
    where shifts in indexes of x and y 
    """
    n = shifts.size
    m = x.size
    cor = np.empty(n, dtype=np.float64)    
 
    for i in range (n):
        ind = shifts[i]
        if shifts[i] < 0:
            cor[i] = stat.pearsonr(x[abs(ind):-1],y[0:ind-1])[0]
        elif  shifts[i] == 0:
            cor[i] = stat.pearsonr(x,y)[0]
        else:
            cor[i] = stat.pearsonr(x[0:m-ind],y[ind:])[0] #np.random.rand(1)#

    return cor
####################################################################
## librarary for filtering    
from scipy.signal import butter, filtfilt


def butter_bandpass(lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)#lfilter(b, a, data)
    return y    
####################################################################
def filtrate_lfp(lfp, fd):
    lfp_mean = np.mean(lfp)
    lfp_std = np.std(lfp)
    lfp_norm = (lfp - lfp_mean) / lfp_std
    
    lfp_fft = 2 * np.fft.rfft(lfp_norm) / lfp.size
    
    w = np.fft.rfftfreq(lfp.size, 1/fd )
    Z = np.exp(-0.05*w)
    Z = 0.8*Z + 0.2
    Z[w>=150] = 0
    
    lfp_fft *= Z
    
    lfp_filtred = (np.fft.irfft(lfp_fft) + lfp_mean)*lfp_std
    
    return lfp_filtred
#####################################################################
def cossfrequency_phase_amp_coupling (signal, fd, ampFrs, phFrs, phasebin):
    import scipy.signal as sig 
    #import matplotlib.pyplot as plt
    w0 = 5    
    phase_signal = butter_bandpass_filter(signal, phFrs.min(), phFrs.max(), fd, 2 )
    coefPh = np.angle( sig.hilbert(phase_signal), deg=False ) 

    #coefPh = theta_phases #
    #plt.plot(coefPh1)
    #plt.plot(coefPh)
    
    scalesAmp = fd*w0/ampFrs/2/np.pi
    # amps_signal = butter_bandpass_filter(signal, ampFrs.min(), ampFrs.max(), fd, 2 )
    coefAmp = np.abs ( mycwt(signal, scalesAmp) )
    
    coupling = np.zeros( [np.ceil(2*np.pi/phasebin).astype(int)+1, ampFrs.size] )

    minPhase = -np.pi #-180.0
    maxPhase = minPhase + phasebin
    ind1 = 0
    while(minPhase <= np.pi ): # 180
        currentPhaseInd = np.logical_and( (coefPh >= minPhase), (coefPh < maxPhase) )
        
        coupling[ind1,:] = np.mean( coefAmp[:, currentPhaseInd ], axis=1 )
        
        ind1 += 1
        minPhase += phasebin
        maxPhase += phasebin

    coupling [np.isnan(coupling)] = 0
    coupling = coupling.transpose()
    return coupling
    
def cossfrequency_phase_phase_coupling (signal, fd, phFrs1, phFrs2, nmarray):

    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    
    signal1 = butter_bandpass_filter(signal, phFrs1.min(), phFrs1.max(), fd, 2 )
    signal2 = butter_bandpass_filter(signal, phFrs2.min(), phFrs2.max(), fd, 2 )
   
    angles1 = np.angle( sig.hilbert(signal1), deg=True ) 
    angles2 = np.angle( sig.hilbert(signal2), deg=True )
    coupling = np.array([])
    
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.hist2d(angles1, angles2, bins=(60, 60), norm=LogNorm())
    fig.suptitle('2d histogram', fontsize=20)
    plt.xlabel('Theta phase', fontsize=16)
    plt.ylabel('Gamma phase', fontsize=16)
    plt.colorbar()
    """
    
    #fig, axarr = plt.subplots(3, 3, subplot_kw=dict(projection='polar'))
    
    #fig2, axarr2 = plt.subplots(3, 3)
    
    for i in range(nmarray.shape[1]):
        ang2 = (angles2 * nmarray[0, i])%(360)
        ang1 = (angles1 * nmarray[1, i])%(360)
        """
        ax2 = axarr2[ (int(i/3))%3, i%3]
        
        ax2.set_title( str(nmarray[0, i]) + ' * theta : ' + str (nmarray[1, i]) + ' * gama' )
        
        H, xedges, yedges, img = ax2.hist2d(ang1-180, ang2-180, bins=(60, 60), norm=LogNorm())     
        plt.title('2d histogram', fontsize=20)
        plt.xlabel('Theta phase', fontsize=16)
        plt.ylabel('Gamma phase', fontsize=16)
        plt.colorbar(img, ax = ax2)
        """
        
        vects_angle = ang2 - ang1
        """
        N = 60
        bottom = 0
        phase_shifts = vects_angle%(360) - 180
        
        theta = np.linspace(0, 2*np.pi, N)
        radii, bin_edges = np.histogram(phase_shifts, N, density=True) 
        width = (2*np.pi) / N
        
        
        ax = axarr[ (int(i/3))%3, i%3]
        # ax = plt.subplot(111, polar=True)
        ax.set_title( str(nmarray[0, i]) + ' * gamma : ' + str (nmarray[1, i]) + ' * theta' )
        bars = ax.bar(theta, radii, width=width, bottom=bottom)
        
        # Use custom colors and opacity
        for r, bar in zip(radii, bars):
            bar.set_facecolor(plt.cm.jet(r / 10.))
            bar.set_alpha(0.8)
        
        #ax2 = axarr2[ (int(i/3))%3, i%3]
        #ax2.hist2d(ang2, ang1, bins=N, norm=LogNorm())
        #ax2.colorbar()
        """
        vects_angle = np.deg2rad(vects_angle)
        x_vect = np.cos( vects_angle )
        y_vect = np.sin( vects_angle )
        mean_resultant_length = np.sqrt( np.sum(x_vect)**2 + np.sum(y_vect)**2 ) / vects_angle.size
        coupling = np.append(coupling, mean_resultant_length)
    #plt.tight_layout()
    #plt.show()
    return (coupling)

def get_asymetry_index(lfp, orders = 25):
    idx_max = argrelmax(lfp, order=orders)[0]
    idx_min = argrelmax(-lfp, order=orders)[0]
    
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
    assymetry_index = assymetry_index[np.logical_not( np.isnan(assymetry_index) ) ]
    return assymetry_index, idx_max, idx_min

def get_units_phase_coupling(lfp, firing, fd):
    angles = np.array([])
    length = np.array([])
    analitic_signal = sig.hilbert(lfp)
    amp = np.abs(analitic_signal)
    amp /= np.linalg.norm(amp) 
    lfp_phases = np.angle(analitic_signal, deg=False )
    
            
    if (firing[1, :].size == 0):
        return 0, 0
        
    start = np.min(firing[1, :]).astype(int)
    end = np.max(firing[1, :]).astype(int) + 1
    for idx in range(start, end):
        spike_times = firing[0, firing[1, :]  == idx]
        if (spike_times.size == 0):
            continue
        spike_ind = np.floor(spike_times*fd).astype(int)
        spike_phases = lfp_phases[spike_ind]
        tmp_x = np.sum( amp[spike_ind] * np.cos(spike_phases) )
        tmp_y = np.sum( amp[spike_ind] * np.sin(spike_phases) )
        mean_phase = np.arctan2(tmp_y, tmp_x)
        length_vect = np.sqrt(tmp_x**2 + tmp_y**2) / spike_phases.size
        angles = np.append(angles, mean_phase)
        length = np.append(length, length_vect)
        
    return angles, length




    