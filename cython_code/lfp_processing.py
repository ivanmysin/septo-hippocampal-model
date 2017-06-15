# -*- coding: utf-8 -*-
"""
main script for processing
"""

import numpy as np
import matplotlib.pyplot as plt
import processingLib as plib
import os
import scipy.signal as sig
from scipy.stats import circmean, circstd

plt.rc('axes', linewidth=2)
plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)
plt.rc('lines', linewidth=3) 
plt.rc('lines', markersize=4)
plt.rc('lines', c="black")

def get_colors():
    neurons_colors = {
        "pyr" : (1.0, 0.2, 0.2), 
        "bas" : (0.4, 1.0, 1.0),
        "olm" : (0.6, 1.0, 0.4),
        "ca3" : (1.0, 0.8, 0.0),
        "ec" :  (1.0, 0.6, 0.6),
        "ms" :  (0.2, 0.6, 1.0),
    }
    return neurons_colors

def make_calculation(path, file):
    Np = 400    # 400    # number of pyramide neurons
    Nb = 50     # number of basket cells
    Nolm = 50   # 50   # number of olm cells
    
    Nec = 100
    Nca3 = 100
    NSG = 100
    
    file = path + file
    
    fd = 10000
    margin = 0.5 # margin from start for analisis in seconds
    
    res = np.load(file)
    

    margin_ind = int(margin * fd)
    V = res[()]["results"]["V"]

    firing = res[()]["results"]["firing"]
    firing[0, :] -= 1000 * margin
    firing = firing[:, firing[0, :] >= 0]
    
    lfp = 0
    for v in V:
        try:
            lfp += v["dendrite"] - v["soma"]
        except KeyError:
            continue
        
    
    lfp = lfp[margin_ind : ] / Np
    lfp_filtred = plib.filtrate_lfp(lfp, 10000)
    lfp_filtred = lfp_filtred[0:-1:20]
    fd_filtered = 500
    
    lfp_filtred_fft = 2 * np.abs( np.fft.rfft(lfp) ) / lfp.size
    freqs = np.fft.rfftfreq(lfp.size, 1/fd)
    
    theta_part = np.sum(lfp_filtred_fft[ (freqs >= 4)&(freqs <= 12) ]) / np.sum(lfp_filtred_fft[1:]) 
    
    theta_lfp = plib.butter_bandpass_filter(lfp_filtred, 4, 10, fd_filtered, 3)
    
    firing_slices = {}

    firing[0, :] *= 0.001
    # plot septal neurons
    cum_it = Np 
    sl = (firing[1, :] < cum_it)
    
    
    firing_slices["pyramide"] = np.copy(sl)

        
    sl = (firing[1, :] > cum_it) & (firing[1, :] <= cum_it + Nb)
    firing_slices["basket"] = np.copy(sl)

    cum_it += Nb
    sl = (firing[1, :] > cum_it) & (firing[1, :] <= cum_it + Nolm)
    firing_slices["olm"] = np.copy(sl)
    
    
    sl = (firing[1, :] > cum_it) & (firing[1, :] <= cum_it + Nolm)
    firing_slices["olm"] = np.copy(sl)
    
    cum_it += Nolm
    sl = (firing[1, :] > cum_it) & (firing[1, :] <= cum_it + NSG)
                
    firing_slices["gaba_pv"] = np.copy(sl)

    cum_it += NSG 
    sl = (firing[1, :] > cum_it) & (firing[1, :] <= cum_it + Nec)    
    firing_slices["ec"] = np.copy(sl)
    cum_it += Nec
    
    sl = (firing[1, :] > cum_it) & (firing[1, :] <= cum_it + Nca3)  

    firing_slices["cs"] = np.copy(sl)
   
    neurons_phases = plib.get_units_disrtibution(theta_lfp, fd_filtered, firing, firing_slices)
    neurons_phases["theta_part"] = theta_part
    
    return neurons_phases
    

def make_figures(path, file, septumInModel): # path, file, septumInModel=True

    Np = 400 # 400    # number of pyramide neurons
    Nb = 50     # number of basket cells
    Nolm = 50  # 50   # number of olm cells
    
    if (septumInModel):
        Nglu = 40
        NgabaCR = 40
        NgabaPV1 = 40 
        NgabaPV2 = 40
        NgabaPacPV1 = 10
        NgabaPacPV2 = 10
    else:
        NSG = 100    # number of spike generators of
    
    Nec = 100
    Nca3 = 100

    neurons_colors = get_colors()

   
    saving_path = path + file[0:2]
    
    file = path + file
    
    fd = 10000
    margin = 0.5 # margin from start for analisis in seconds
    
    res = np.load(file)
    
    lfp = res[()]["results"]["lfp"]
    margin_ind = int(margin * fd)
    lfp = lfp[margin_ind : ]
    V = res[()]["results"]["V"]
    currents = res[()]["results"]["currents"]
    firing = res[()]["results"]["firing"]
    firing[0, :] -= 1000 * margin
    firing = firing[:, firing[0, :] >= 0]
    
    
    
    lfp_filtred = plib.filtrate_lfp(lfp, fd)
    lfp_filtred = lfp_filtred[0:-1:20]
    fd = 500
    t = np.linspace(0, lfp_filtred.size/fd, lfp_filtred.size)

    
    lfp2 = 0
    Vdm = -60
    Vsm = -60
    for v in V:
        try:
            Vdm += v["dendrite"]
            Vsm += v["soma"]
            lfp2 += v["dendrite"] - v["soma"]
        except KeyError:
            continue
    # lfp *= 1000 # recalculate from mV to mkV 
    # lfp2 *= 1000 # recalculate from mV to mkV 
    Vdm = Vdm[margin_ind : ] / Np
    Vsm = Vsm[margin_ind : ] / Np
        
    lfp2 = lfp2[margin_ind : ] / Np
    lfp_filtred2 = plib.filtrate_lfp(lfp2, 10000)
    lfp_filtred2 = lfp_filtred2[0:-1:20]
    
    pyr_firing_max_idx = 0
    tmp_count = 0
    for idx in range(Np):
        tmp = np.count_nonzero(firing[1, :] == idx)
        if (tmp > tmp_count):
            tmp_count = tmp
            pyr_firing_max_idx = idx - 1
    

    bas_firing_max_idx = Np + 1
    tmp_count = 0
    for idx in range(Np, Np + Nb):
        tmp = np.count_nonzero(firing[1, :] == idx)
        if (tmp > tmp_count):
            tmp_count = tmp
            bas_firing_max_idx = idx - 1
    
    olm_firing_max_idx = Np + Nb + 1
    tmp_count = 0
    for idx in range( Np + Nb,  Np + Nb + Nolm):
        tmp = np.count_nonzero(firing[1, :] == idx)
        if (tmp > tmp_count):
            tmp_count = tmp
            olm_firing_max_idx = idx - 1
    
    
    Vpyr = V[pyr_firing_max_idx]["soma"][margin_ind : ] - 60
    # Vpyrd = V[pyr_firing_max_idx]["dendrite"][margin_ind : ] - 60
    Vbas = V[bas_firing_max_idx]["soma"][margin_ind : ]
    Volm = V[olm_firing_max_idx]["soma"][margin_ind : ]
    
    
    t4v = np.linspace(0, t[-1], Vpyr.size)
    
    
    
    plt.figure( figsize = (10, 5), tight_layout=True)
    plt.subplot(311)
    plt.plot(t4v, Vpyr, color=neurons_colors["pyr"])
    plt.xlim(0, 1.5)
    plt.ylim(-95, 70)

    
    plt.subplot(312)
    plt.plot(t4v, Vbas, color=neurons_colors["bas"])
    plt.xlim(0, 1.5)
    plt.ylim(-95, 70)

    
    plt.subplot(313)
    plt.plot(t4v, Volm, color=neurons_colors["olm"])
    plt.xlim(0, 1.5)
    plt.ylim(-95, 70)
    
    
    plt.savefig(saving_path + "Vhist.png", dpi = 500)
    
    
    
    plt.figure( figsize = (10, 2), tight_layout=True )
    #plt.subplot(211)
    plt.plot(t, lfp_filtred2, color="black")
    plt.xlim(0, 1.5)
    # plt.subplot(212)
    # plt.plot(t, lfp_filtred, "g")
    plt.savefig(saving_path + "lfp2.png", dpi = 500)
    
    lfp = lfp2 # !!!!!!!!!!
    lfp_filtred = lfp_filtred2 # !!!!!!!!!!!!!!!!
    
    # calculate and plot wavelet spectrum and LFP signal
    freqs, coefs = plib.computemycwt(fd, lfp_filtred)
    
    
    plt.figure( figsize = (10, 5), tight_layout=True  )
    plt.subplot(211)
    Z = np.abs(coefs) * 1000
    plt.pcolor(t, freqs, Z, cmap='rainbow', vmin=Z.min(), vmax= Z.max())
    plt.title('Wavelet spectrum of simulated LFP')
    # set the limits of the plot to the limits of the data
    plt.axis([t[0], t[-1], freqs[0], 30])
    plt.colorbar()
    
    
    plt.subplot(212)
    plt.plot(t, 1000 * lfp_filtred, color="black")
    plt.xlim(t[0], t[-1])
    # plt.ylim(1.2*lfp_filtred.max(), -1.2*lfp_filtred.min())
    plt.colorbar()
    plt.savefig(saving_path + "wavelet.png", dpi = 500)
    
    
    lfp_filtred_fft = 2 * np.abs( np.fft.rfft(lfp2) ) / lfp2.size
    freqs = np.fft.rfftfreq(lfp2.size, 1/10000)
    
    theta_part = np.sum(lfp_filtred_fft[ (freqs >= 4)&(freqs <= 12) ]) / np.sum(lfp_filtred_fft[1:]) 
    
    
    plt.figure( tight_layout=True )
    plt.plot(freqs[1:], lfp_filtred_fft[1:])
    plt.xlim(0, 20)
    plt.ylim(0, 0.1)
    plt.savefig(saving_path + "fft_lfp_filteterd.png", dpi = 500)
    
    
    
    
    # calculate and plot asymmetry index
    asymmetry_index, idx_max, idx_min = plib.get_asymetry_index(lfp_filtred)
    if (asymmetry_index.size > 0):
        plt.figure()
        plt.subplot(121)
        plt.plot(t, lfp_filtred, color="black", linewidth=2)
        lfp_max = lfp_filtred.max()
        lfp_min = lfp_filtred.min()
        for i in idx_max:
            plt.plot([t[i], t[i]], [lfp_min, lfp_max], color="blue", linestyle="dashed", linewidth=2)
    
        for i in idx_min:
            plt.plot([t[i], t[i]], [lfp_min, lfp_max], color="blue", linestyle="dashed", linewidth=2)
            
        plt.subplot(122)    
        plt.hist(asymmetry_index, 10, normed=1, facecolor='blue', alpha=0.75)
        plt.xlim(-1.5, 1.5)
        plt.xlabel('Asymmetry index')
        plt.ylabel('Theta cycles')
        plt.title('')
        plt.savefig(saving_path + "asymmetry_index.png", dpi = 500)
    else:
        print ("Zero elements in asymemetry index array")
    
    # calculate and plot amplitude - phase cross frequency coupling between theta and gamma rhythms
    ampFrs = np.linspace(30, 80, 50)
    phFrs = np.array([4.0, 10.0])
    apcoupling = plib.cossfrequency_phase_amp_coupling(lfp_filtred, fd, ampFrs, phFrs, 0.1)
    theta_phase = np.linspace(-np.pi, np.pi, apcoupling.shape[1])
    plt.figure()
    plt.subplot(111)
    plt.pcolor(theta_phase, ampFrs, apcoupling, cmap='rainbow', vmin=apcoupling.min(), vmax=apcoupling.max())
    plt.title('Cross frequency coupling between theta and gamma rhythms')
    plt.axis([-np.pi, np.pi, ampFrs[0], ampFrs[-1]])
    plt.colorbar()
    plt.savefig(saving_path + "amp_phase_coupling.png", dpi = 500)
    
    # calculate and plot phase-phase cross frequency coupling between theta and gamma rhythms
    phFrs1 = np.array([4.0, 10.0])
    phFrs2 = np.array([30, 90])
    nmarray = np.ones([2, 12])
    nmarray[1, :] = np.arange(1, 13)
    ppcoupling = plib.cossfrequency_phase_phase_coupling (lfp_filtred, fd, phFrs1, phFrs2, nmarray)
    
    plt.figure()
    plt.plot(nmarray[1, :], ppcoupling)
    plt.xlim(1, nmarray[1, -1])
    plt.savefig(saving_path + "phase_phase_coupling.png", dpi = 500)
# 
    
    # calculate and plot phase coupling between units activity and theta rhythm 
    
    
    theta_lfp = plib.butter_bandpass_filter(lfp_filtred, 4, 10, fd, 3)
    
    plt.figure( figsize = (10, 2), tight_layout=True  )
    plt.plot(t, theta_lfp, color="black")
    plt.xlim(t[0], t[-1])
    plt.savefig(saving_path + "theta_lfp.png", dpi = 500)
    
    
    
    
    
    
    plt.figure()
    plt.plot(t, lfp_filtred, color="black")
    plt.xlim(0, 1.5) # plt.xlim(t[0], t[-1])
    
    
    # plt.subplot(312)
    # plt.plot(t, theta_lfp, color="blue")
    # plt.xlim(0, 1.5) #plt.xlim(t[0], t[-1])
    firing_slices = {}
    # plt.figure( figsize = (10, 5), tight_layout=True   )
    fig, (a1, a2, a3, a4, a5, a6) = plt.subplots(6, 1, sharex=True, gridspec_kw = {'height_ratios':[2, 2, 2, 1, 1, 8]},  figsize = (10, 5))
    fig.set_size_inches(10, 10)
    
    
    
    firing[0, :] *= 0.001
    # plot septal neurons
    if (septumInModel):
        pass
#        cum_it = Nglu
#        sl = firing[1, :] <= cum_it
#        firing_slices["glu"] = np.copy(sl)
#        glu_line = plt.scatter(firing[0, sl], firing[1, sl], color="r")
#        
#        sl = (firing[1, :] > cum_it) & (firing[1, :] <= cum_it + NgabaCR)
#        firing_slices["gaba_cr"] = np.copy(sl)
#        cr_line = plt.scatter(firing[0, sl], firing[1, sl], color="g")
#        cum_it += NgabaCR
#        
#        sl = (firing[1, :] > cum_it) & (firing[1, :] <= cum_it + NgabaPV1 + NgabaPacPV1)
#        firing_slices["gaba_pv1"] = np.copy(sl)
#        pv1_line = plt.scatter(firing[0, sl], firing[1, sl], color="b")
#        cum_it += NgabaPV1 + NgabaPacPV1
#        
#        sl = (firing[1, :] > cum_it) & (firing[1, :] <= cum_it + NgabaPV2 + NgabaPacPV2)
#        firing_slices["gaba_pv2"] = np.copy(sl)
#        pv2_line = plt.scatter(firing[0, sl], firing[1, sl], color="b")
#        cum_it += NgabaPV2 + NgabaPacPV2
#
#        # plot hippocampal neurons
#        sl = (firing[1, :] > cum_it) & (firing[1, :] <= cum_it + Np)
    else:
        cum_it = Np 
        sl = (firing[1, :] < cum_it)
    
    
    firing_slices["pyramide"] = np.copy(sl)
    pyr_line = a6.scatter(firing[0, sl], firing[1, sl], color=neurons_colors["pyr"])
    
    if (septumInModel):
        cum_it += Np
    
        
    sl = (firing[1, :] > cum_it) & (firing[1, :] <= cum_it + Nb)
    firing_slices["basket"] = np.copy(sl)
    basket_line = a5.scatter(firing[0, sl], firing[1, sl], color=neurons_colors["bas"])
    a5.set_ylim(400, 450)
    cum_it += Nb
    sl = (firing[1, :] > cum_it) & (firing[1, :] <= cum_it + Nolm)
    firing_slices["olm"] = np.copy(sl)
    olm_line = a4.scatter(firing[0, sl], firing[1, sl], color=neurons_colors["olm"])
    
    sl = (firing[1, :] > cum_it) & (firing[1, :] <= cum_it + Nolm)
    firing_slices["olm"] = np.copy(sl)
    
    if not(septumInModel):
        cum_it += Nolm
        sl = (firing[1, :] > cum_it) & (firing[1, :] <= cum_it + NSG)
                
        pv_line = a3.scatter(firing[0, sl], firing[1, sl], color=neurons_colors["ms"])
        firing_slices["gaba_pv"] = np.copy(sl)
    
#    if (septumInModel):          
#        plt.legend((glu_line, cr_line, pv1_line, pv2_line, pyr_line, basket_line, olm_line),
#               ('Glu', 'GABA(CR)', 'GABA(PV1)', 'GABA(PV2)', 'Pyramide', 'Basket', 'OLM'),
#               scatterpoints=1,
#               loc='upper left',
#               ncol=1,
#               fontsize=12)
#    else:
#        plt.legend((pv_line, pyr_line, basket_line, olm_line),
#               ('GABA(PV)',  'Pyramide', 'Basket', 'OLM'),
#               scatterpoints=1,
#               loc='upper left',
#               ncol=1,
#               fontsize=12)
    
    cum_it += NSG 
    sl = (firing[1, :] > cum_it) & (firing[1, :] <= cum_it + Nec)    
    a2.scatter(firing[0, sl], firing[1, sl], color=neurons_colors["ec"])
    firing_slices["ec"] = np.copy(sl)
    cum_it += Nec
    
    sl = (firing[1, :] > cum_it) & (firing[1, :] <= cum_it + Nca3)  
    a1.scatter(firing[0, sl], firing[1, sl], color=neurons_colors["ca3"])
    firing_slices["cs"] = np.copy(sl)
    
    plt.xlim(0, 1.5) #(0, t[-1])
    # plt.ylim(0, 800)
    
    fig.tight_layout()
    fig.savefig(saving_path + "raster.png", dpi = 500)
    
  
    
    
    neurons_phases_for_return = plib.get_units_disrtibution(theta_lfp, fd, firing, firing_slices)
    
    neurons_phases = {}
    for neuron_phase_key in neurons_phases_for_return.keys():
        neurons_phases[neuron_phase_key] = np.append(neurons_phases_for_return[neuron_phase_key], neurons_phases_for_return[neuron_phase_key] + 2*np.pi )
    
    phases_x = np.linspace(-np.pi, 3*np.pi, 40)
    phases_y = 0.5*np.cos(phases_x) + 0.5
    plt.figure( tight_layout=True )
    plt.subplot(311)

    plt.hist(neurons_phases["pyramide"], 40, normed=True, facecolor=neurons_colors["pyr"], alpha=0.75)
    plt.plot(phases_x, phases_y, color="black")
    plt.xlim(-np.pi, 3*np.pi)
    plt.subplot(312)
    plt.hist(neurons_phases["basket"], 40, normed=True, facecolor=neurons_colors["bas"], alpha=0.75)
    plt.plot(phases_x, phases_y, color="black")
    plt.xlim(-np.pi, 3*np.pi)
    plt.subplot(313)
    plt.hist(neurons_phases["olm"], 40, normed=True, facecolor=neurons_colors["olm"], alpha=0.75)
    plt.plot(phases_x, phases_y, color="black")
    plt.xlim(-np.pi, 3*np.pi)
    plt.tight_layout()
    plt.savefig(saving_path + "phase_disribution_histogram.png", dpi = 500)
    
    plt.figure()
    plt.subplot(111, polar=True)

    
    color = "y"
    for key, sl in firing_slices.items():
        fir = firing[:, sl]
        if (fir.size == 0):
            continue
        
        
        angles, length = plib.get_units_phase_coupling(theta_lfp, fir, fd)
        #angles += np.pi/2
        
        if (key == "glu"):
            color = "r"
        if (key == "gaba_cr"):
            color = "g"
        if (key == "gaba_pv1" or key == "gaba_pv2" or key == "gaba_pv"):
            color = neurons_colors["ms"]
        if (key == "pyramide"):
            color = neurons_colors["pyr"]
        if (key == "basket"):
            color = neurons_colors["bas"]  
        if (key == "olm"):
            color = neurons_colors["olm"]
        if (key == "cs"):
            color = neurons_colors["ca3"]
        if (key == "ec"):
            color = neurons_colors["ec"]
        
        plt.scatter(angles, length, color=color)
        
        # calculate histogram ?
    

    plt.savefig(saving_path + "phase_disribution_of_neurons.png", dpi = 500)
    plt.figure( tight_layout=True )
    plt.subplot(111)
    #theta_lfp = plib.butter_bandpass_filter(lfp_filtred, 4, 12, fd, 2)
    
    color = "g"
    for key, sl in firing_slices.items():
        
        fir = firing[:, sl]
        if (fir.size == 0):
            continue
        angles, length = plib.get_units_phase_coupling(theta_lfp, fir, fd)
        #angles += np.pi/2
        minn = np.min( firing[1, sl] ) 
        maxn = np.max( firing[1, sl] ) 
        numbers = np.linspace(minn, maxn, angles.size)
        
        if (key == "glu"):
            color = "r"
        if (key == "gaba_cr"):
            color = "g"
        if (key == "gaba_pv1" or key == "gaba_pv2" or key == "gaba_pv"):
            color = neurons_colors["ms"]
        if (key == "pyramide"):
            color = neurons_colors["pyr"]
        if (key == "basket"):
            color = neurons_colors["bas"]  
        if (key == "olm"):
            color = neurons_colors["olm"]
        if (key == "cs"):
            color = neurons_colors["ca3"]
        if (key == "ec"):
            color = neurons_colors["ec"]
        plt.scatter(angles, numbers, color=color)

        # calculate histogram ?
    plt.ylim(0, 800)
    tmp_phases = np.linspace(-np.pi, np.pi, 1000)
    plt.plot( tmp_phases, 200 * ( np.cos(tmp_phases) + 1 ), color="black" )
    plt.xlim(-np.pi, np.pi)
    
    plt.savefig(saving_path + "phase_disribution_of_neurons2.png", dpi = 500)
    
    

    
    
    soma_currents = np.zeros(currents[0]["soma"].size)
    dendrite_current = np.zeros_like(soma_currents)
    Vin = np.zeros(V[0]["soma"].size)
    for cur, vin in zip(currents, V):
        if "dendrite" in cur.keys():
            soma_currents += cur["soma"]
            dendrite_current += cur["dendrite"]
            Vin += vin["soma"]
    
    soma_currents = sig.resample(soma_currents[margin_ind: ], lfp_filtred.size) 
    dendrite_current = sig.resample(dendrite_current[margin_ind : ], lfp_filtred.size) 
    Vin /= Np
    Vin = sig.resample(Vin[margin_ind: ], lfp_filtred.size) 
    
    
    soma_currents = (soma_currents - soma_currents.mean() ) / soma_currents.std()
    dendrite_current = (dendrite_current - dendrite_current.mean() ) / dendrite_current.std()
    lfp_filtred_normed = (lfp_filtred - lfp_filtred.mean()) / lfp_filtred.std()
    plt.figure( tight_layout=True, figsize=(8, 5) )
    plt.subplot(311)
    plt.plot(t, Vin)
    plt.title("Intracellular potential on soma")
    plt.xlim(0, 1.5) #plt.xlim(1, 1.5)
    plt.subplot(312)
    plt.plot(t, soma_currents, "b")
    plt.plot(t, lfp_filtred_normed, "g")
    #plt.ylim(-1, 1)
    plt.xlim(0, 1.5) # plt.xlim(1, 1.5) #
    plt.title("soma")
    plt.subplot(313)
    plt.plot(t, dendrite_current, "b")
    plt.plot(t, lfp_filtred_normed, "g")
    plt.xlim(0, 1.5) # plt.xlim(1, 1.5) #
    # plt.ylim(-1, 1)
    plt.title("dendrite")
#    plt.subplot(414)    
#    # plot septal neurons
#    if (septumInModel):
#        cum_it = Nglu
#        sl = firing[1, :] <= cum_it
#        firing_slices["glu"] = np.copy(sl)
#        glu_line = plt.scatter(firing[0, sl], firing[1, sl], color="r")
#        
#        sl = (firing[1, :] > cum_it) & (firing[1, :] <= cum_it + NgabaCR)
#        firing_slices["gaba_cr"] = np.copy(sl)
#        cr_line = plt.scatter(firing[0, sl], firing[1, sl], color="g")
#        cum_it += NgabaCR
#        
#        sl = (firing[1, :] > cum_it) & (firing[1, :] <= cum_it + NgabaPV1 + NgabaPacPV1)
#        firing_slices["gaba_pv1"] = np.copy(sl)
#        pv1_line = plt.scatter(firing[0, sl], firing[1, sl], color="b")
#        cum_it += NgabaPV1 + NgabaPacPV1
#        
#        sl = (firing[1, :] > cum_it) & (firing[1, :] <= cum_it + NgabaPV2 + NgabaPacPV2)
#        firing_slices["gaba_pv2"] = np.copy(sl)
#        pv2_line = plt.scatter(firing[0, sl], firing[1, sl], color="b")
#        cum_it += NgabaPV2 + NgabaPacPV2
#
#        # plot hippocampal neurons
#        sl = (firing[1, :] > cum_it) & (firing[1, :] <= cum_it + Np)
#    else:
#        cum_it = Np 
#        sl = (firing[1, :] < cum_it)
#    firing_slices["pyramide"] = np.copy(sl)
#    pyr_line = plt.scatter(firing[0, sl], firing[1, sl], color="c")
#    
#    if (septumInModel):
#        cum_it += Np
#    
#        
#    sl = (firing[1, :] > cum_it) & (firing[1, :] <= cum_it + Nb)
#    firing_slices["basket"] = np.copy(sl)
#    basket_line = plt.scatter(firing[0, sl], firing[1, sl], color="k")
#    cum_it += Nb
#    sl = (firing[1, :] > cum_it) & (firing[1, :] <= cum_it + Nolm)
#    firing_slices["olm"] = np.copy(sl)
#    olm_line = plt.scatter(firing[0, sl], firing[1, sl], color="m")
#    plt.ylim(0, 800)
#    #plt.xlim(1, 1.5) #plt.xlim(0, 1.5)
#    if not(septumInModel):
#        cum_it += Nolm
#        sl = (firing[1, :] > cum_it)
#                
#        pv_line = plt.scatter(firing[0, sl], firing[1, sl], color="b")
#        firing_slices["gaba_pv1"] = np.copy(sl)
#    
#    if (septumInModel):          
#        plt.legend((glu_line, cr_line, pv1_line, pv2_line, pyr_line, basket_line, olm_line),
#               ('Glu', 'GABA(CR)', 'GABA(PV1)', 'GABA(PV2)', 'Pyramide', 'Basket', 'OLM'),
#               scatterpoints=1,
#               loc='upper left',
#               ncol=1,
#               fontsize=8)
#    else:
#        plt.legend((pv_line, pyr_line, basket_line, olm_line),
#               ('GABA(PV)',  'Pyramide', 'Basket', 'OLM'),
#               scatterpoints=1,
#               loc='upper left',
#               ncol=1,
#               fontsize=8)
    

    plt.savefig(saving_path + "currents.png", dpi = 500)
    plt.close('all')
    
    neurons_phases_for_return["theta_part"] = theta_part
    return neurons_phases_for_return
     
################################################################
def ms_cs_pp_phase_shift_processing(processed_data, saving_path):
    
    
    
    
    phase_shift = np.linspace(-np.pi, np.pi, 30)
    
    neurons_colors = get_colors()
    
    data_mean_pyr = []
    data_varience_pyr = []
    
    data_mean_bas = []
    data_varience_bas = []

    data_mean_olm = []
    data_varience_olm = []
    
    
    new_processed_data = []
    
    for idx1 in range(30):
        
        
        pyr_data = np.array([], dtype=float)
        bas_data = np.array([], dtype=float)
        olm_data = np.array([], dtype=float)
        theta_part = np.array([], dtype=float)
        
        
        for t in processed_data[idx1:-1:30]:
            # print (t)
            pyr_data = np.append(pyr_data, t["pyramide"])
            bas_data = np.append(bas_data, t["basket"])
            olm_data = np.append(olm_data, t["olm"])
            theta_part = np.append(theta_part, t["theta_part"])
            
        tmp_dict = {"pyramide":pyr_data, "basket":bas_data, "olm":olm_data, "theta_part":theta_part}
        
        new_processed_data.append(tmp_dict)

    theta_part = []
    for data_value in new_processed_data:
        data_mean_pyr.append( circmean(data_value["pyramide"], high=np.pi, low=-np.pi) )
        data_varience_pyr.append( circstd(data_value["pyramide"], high=np.pi, low=-np.pi) )
        
        data_mean_bas.append( circmean(data_value["basket"], high=np.pi, low=-np.pi) )
        data_varience_bas.append( circstd(data_value["basket"], high=np.pi, low=-np.pi) )
        
        data_mean_olm.append( circmean(data_value["olm"], high=np.pi, low=-np.pi) )
        data_varience_olm.append( circstd(data_value["olm"], high=np.pi, low=-np.pi) )
        
        theta_part.append(data_value["theta_part"])
        
    phases = np.linspace(-2*np.pi, 2*np.pi, 100)
    wave = np.cos(phases)
    fig, axs = plt.subplots(nrows=1, ncols=3, sharex=True)
    axs[0].plot(phases, wave, color="black", linewidth=2)
    axs[0].errorbar(data_mean_pyr, phase_shift, xerr=data_varience_pyr, fmt='o', color=neurons_colors["pyr"] )
    
    axs[0].set_xlim(-2*np.pi, 2*np.pi)
    axs[0].set_title('Pyr')
    axs[1].plot(phases, wave, color="black", linewidth=2)
    axs[1].errorbar(data_mean_bas, phase_shift, xerr=data_varience_bas, fmt='o', color=neurons_colors["bas"])
    
    axs[1].set_xlim(-2*np.pi, 2*np.pi)
    axs[1].set_title('Bas')
    
    axs[2].plot(phases, wave, color="black", linewidth=2)
    axs[2].errorbar(data_mean_olm, phase_shift, xerr=data_varience_olm, fmt='o', color=neurons_colors["olm"])
    axs[2].set_xlim(-2*np.pi, 2*np.pi)
    axs[2].set_title('OLM')
    
    fig.tight_layout()
    fig.savefig(saving_path + "ms_cs_pp_phase_shift.png", dpi=500)

    
    theta_part_mean = np.zeros(30, dtype=float)
    theta_part_std = np.zeros(30, dtype=float)
    for idx, t in enumerate(theta_part):
        theta_part_mean[idx] = np.mean(t)
        theta_part_std[idx] = np.std(t)
    fig, axs = plt.subplots()
    fig.set_size_inches(10, 5)
    axs.errorbar(phase_shift, theta_part_mean, yerr=theta_part_std, fmt='o')
    
    
    fig.savefig(saving_path + "ms_cs_pp_phase_shift_theta_power.png", dpi=500)
    
    
def without_of_each_input_processing(processed_data, control_data, saving_path):
    
    neurons_colors = get_colors()
    
    without_pv1_pyr = np.array([], dtype=float)
    without_pv1_bas = np.array([], dtype=float)
    without_pv1_olm = np.array([], dtype=float)
    
    without_pv2_pyr = np.array([], dtype=float)
    without_pv2_bas = np.array([], dtype=float)
    without_pv2_olm = np.array([], dtype=float)
    
    without_pp_pyr = np.array([], dtype=float)
    without_pp_bas = np.array([], dtype=float)
    without_pp_olm = np.array([], dtype=float)
    
    without_cs_pyr = np.array([], dtype=float)
    without_cs_bas = np.array([], dtype=float)
    without_cs_olm = np.array([], dtype=float)
    
    without_ms_pyr = np.array([], dtype=float)
    without_ms_bas = np.array([], dtype=float)
    without_ms_olm = np.array([], dtype=float)
    
    without_ex_pyr = np.array([], dtype=float)
    without_ex_bas = np.array([], dtype=float)
    without_ex_olm = np.array([], dtype=float)
    
    theta_part = np.array([], dtype=float)
    
    for data_value in control_data:
        theta_part = np.append( theta_part, data_value["theta_part"] )
    
    for idx, data_value in enumerate(processed_data):
        theta_part = np.append( theta_part, data_value["theta_part"] )
        
        
        if (idx//10 == 0):
            without_pv1_pyr = np.append(without_pv1_pyr, data_value["pyramide"])
            without_pv1_bas = np.append(without_pv1_bas, data_value["basket"])
            without_pv1_olm = np.append(without_pv1_olm, data_value["olm"])
            
            
                      
        if (idx//10 == 1):
            without_pv2_pyr = np.append(without_pv2_pyr, data_value["pyramide"])
            without_pv2_bas = np.append(without_pv2_bas, data_value["basket"])
            without_pv2_olm = np.append(without_pv2_olm, data_value["olm"])
           
        
        if (idx//10 == 2):
            without_pp_pyr = np.append(without_pp_pyr, data_value["pyramide"])
            without_pp_bas = np.append(without_pp_bas, data_value["basket"])
            without_pp_olm = np.append(without_pp_olm, data_value["olm"])

        
        if (idx//10 == 3):
            without_cs_pyr = np.append(without_cs_pyr, data_value["pyramide"])
            without_cs_bas = np.append(without_cs_bas, data_value["basket"])
            without_cs_olm = np.append(without_cs_olm, data_value["olm"])
            
            
        if (idx//10 == 4):
            without_ms_pyr = np.append(without_ms_pyr, data_value["pyramide"])
            without_ms_bas = np.append(without_ms_bas, data_value["basket"])
            without_ms_olm = np.append(without_ms_olm, data_value["olm"])
        
        if (idx//10 == 5):
            without_ex_pyr = np.append(without_ex_pyr, data_value["pyramide"])
            without_ex_bas = np.append(without_ex_bas, data_value["basket"])
            without_ex_olm = np.append(without_ex_olm, data_value["olm"])   
            
            
    
    without_pv1_pyr = np.append(without_pv1_pyr, without_pv1_pyr + 2*np.pi)
    without_pv1_bas = np.append(without_pv1_bas, without_pv1_bas + 2*np.pi)
    without_pv1_olm = np.append(without_pv1_olm, without_pv1_olm + 2*np.pi)
    
    without_pv2_pyr = np.append(without_pv2_pyr, without_pv2_pyr + 2*np.pi)
    without_pv2_bas = np.append(without_pv2_bas, without_pv2_bas + 2*np.pi)
    without_pv2_olm = np.append(without_pv2_olm, without_pv2_olm + 2*np.pi)
    
    without_pp_pyr = np.append(without_pp_pyr, without_pp_pyr + 2*np.pi)
    without_pp_bas = np.append(without_pp_bas, without_pp_bas + 2*np.pi)
    without_pp_olm = np.append(without_pp_olm, without_pp_olm + 2*np.pi)
    
    without_cs_pyr = np.append(without_cs_pyr, without_cs_pyr + 2*np.pi)
    without_cs_bas = np.append(without_cs_bas, without_cs_bas + 2*np.pi)
    without_cs_olm = np.append(without_cs_olm, without_cs_olm + 2*np.pi)
    
    without_ms_pyr = np.append(without_ms_pyr, without_ms_pyr + 2*np.pi)
    without_ms_bas = np.append(without_ms_bas, without_ms_bas + 2*np.pi)
    without_ms_olm = np.append(without_ms_olm, without_ms_olm + 2*np.pi)    
    
    without_ex_pyr = np.append(without_ex_pyr, without_ex_pyr + 2*np.pi)
    without_ex_bas = np.append(without_ex_bas, without_ex_bas + 2*np.pi)
    without_ex_olm = np.append(without_ex_olm, without_ex_olm + 2*np.pi)      
    
    
    
    
    fig, axs = plt.subplots(nrows=3, ncols=6, figsize=(20, 10), tight_layout=True, sharex=True, sharey=True)
    axs[0, 0].hist(without_pv1_pyr, bins=40, normed=True, alpha=0.75, color=neurons_colors["pyr"])
    axs[1, 0].hist(without_pv1_bas, bins=40, normed=True, alpha=0.75, color=neurons_colors["bas"])
    axs[2, 0].hist(without_pv1_olm, bins=40, normed=True, alpha=0.75, color=neurons_colors["olm"])
    
    axs[0, 1].hist(without_pv2_pyr, bins=40, normed=True, alpha=0.75, color=neurons_colors["pyr"])
    axs[1, 1].hist(without_pv2_bas, bins=40, normed=True, alpha=0.75, color=neurons_colors["bas"])
    axs[2, 1].hist(without_pv2_olm, bins=40, normed=True, alpha=0.75, color=neurons_colors["olm"])
    
    axs[0, 2].hist(without_pp_pyr, bins=40, normed=True, alpha=0.75, color=neurons_colors["pyr"])
    axs[1, 2].hist(without_pp_bas, bins=40, normed=True, alpha=0.75, color=neurons_colors["bas"])
    axs[2, 2].hist(without_pp_olm, bins=40, normed=True, alpha=0.75, color=neurons_colors["olm"])

    axs[0, 3].hist(without_cs_pyr, bins=40, normed=True, alpha=0.75, color=neurons_colors["pyr"])
    axs[1, 3].hist(without_cs_bas, bins=40, normed=True, alpha=0.75, color=neurons_colors["bas"])
    axs[2, 3].hist(without_cs_olm, bins=40, normed=True, alpha=0.75, color=neurons_colors["olm"])
    
    
    axs[0, 4].hist(without_ms_pyr, bins=40, normed=True, alpha=0.75, color=neurons_colors["pyr"])
    axs[1, 4].hist(without_ms_bas, bins=40, normed=True, alpha=0.75, color=neurons_colors["bas"])
    axs[2, 4].hist(without_ms_olm, bins=40, normed=True, alpha=0.75, color=neurons_colors["olm"])
    
    axs[0, 5].hist(without_ex_pyr, bins=40, normed=True, alpha=0.75, color=neurons_colors["pyr"])
    axs[1, 5].hist(without_ex_bas, bins=40, normed=True, alpha=0.75, color=neurons_colors["bas"])
    axs[2, 5].hist(without_ex_olm, bins=40, normed=True, alpha=0.75, color=neurons_colors["olm"])
    
    
    
    phases = np.linspace(-np.pi, 3*np.pi, 200)
    signal = 0.25 * (np.cos(phases) + 1)
    
    for idx in range(3):
        for ax in axs[idx]:
            ax.plot(phases, signal, color="black", linewidth=2)
            ax.set_xlim(-np.pi, 3*np.pi)
            ax.set_ylim(0, 0.5)
            ax.xaxis.set_tick_params(labelsize=36)
            ax.yaxis.set_tick_params(labelsize=36)
    fig.savefig(saving_path + "without_any_inputs.png", dpi=500)
    
    
    
    theta_part = theta_part.reshape(7, 10).T
    fig = plt.figure()
    plt.boxplot(theta_part, sym="")
    # plt.ylim(0, 0.2)
    fig.savefig(saving_path + "without_any_inputs_theta_power.png", dpi=500)
    

def pv1_pv2_phase_shift(processed_data, saving_path):
    
    pyr_2_15 = np.array([], dtype=float)
    bas_2_15 = np.array([], dtype=float)
    olm_2_15 = np.array([], dtype=float)

    pyr_pi = np.array([], dtype=float)
    bas_pi = np.array([], dtype=float)
    olm_pi = np.array([], dtype=float)

    for idx, data in enumerate(processed_data):
        if idx < 5:
            pyr_2_15 = np.append(pyr_2_15, data["pyramide"])
            bas_2_15 = np.append(bas_2_15, data["basket"])
            olm_2_15 = np.append(olm_2_15, data["olm"])
        else:
            pyr_pi = np.append(pyr_pi, data["pyramide"])
            bas_pi = np.append(bas_pi, data["basket"])
            olm_pi = np.append(olm_pi, data["olm"])
            
    pyr_2_15 = np.append(pyr_2_15, pyr_2_15 + 2*np.pi)
    bas_2_15 = np.append(bas_2_15, bas_2_15 + 2*np.pi)
    olm_2_15 = np.append(olm_2_15, olm_2_15 + 2*np.pi)
    
    pyr_pi = np.append(pyr_pi, pyr_pi + 2*np.pi)
    bas_pi = np.append(bas_pi, bas_pi + 2*np.pi)
    olm_pi = np.append(olm_pi, olm_pi + 2*np.pi)    
    
    
    
    fig, axs = plt.subplots(nrows=3, ncols=2, sharex=True)
    axs[0, 0].hist(pyr_2_15, bins=40, normed=True, alpha=0.75)
    axs[1, 0].hist(bas_2_15, bins=40, normed=True, alpha=0.75)
    axs[2, 0].hist(olm_2_15, bins=40, normed=True, alpha=0.75)
    
    axs[0, 1].hist(pyr_pi, bins=40, normed=True, alpha=0.75)
    axs[1, 1].hist(bas_pi, bins=40, normed=True, alpha=0.75)
    axs[2, 1].hist(olm_pi, bins=40, normed=True, alpha=0.75)
    
    
    phases = np.linspace(-np.pi, 3*np.pi, 200)
    signal = 0.25 * (np.cos(phases) + 1)
    
    for idx in range(3):
        for ax in axs[idx]:
            ax.plot(phases, signal, linewidth=2)
            ax.set_xlim(-np.pi, 3*np.pi)
            ax.set_ylim(0, 0.5)

        
    fig.tight_layout()
    fig.savefig(saving_path + "pv1-pv2_phase_shift.png", dpi=500)
            

def balance_cs_bas2pyr(processed_data, saving_path):
    
    number_synapses = np.arange(0, 110, 10, dtype=int)
    
    data_mean_pyr = []
    data_varience_pyr = []
    
    data_mean_bas = []
    data_varience_bas = []

    data_mean_olm = []
    data_varience_olm = []
    
    theta_part = []
    
    pyr_tmp = np.array([], dtype=float)
    bas_tmp = np.array([], dtype=float)
    olm_tmp = np.array([], dtype=float)
    theta_tmp = np.array([], dtype=float)
    
    for idx, data_value in enumerate(processed_data):
       
        
        pyr_tmp = np.append(pyr_tmp, data_value["pyramide"])
        bas_tmp = np.append(bas_tmp, data_value["basket"])
        olm_tmp = np.append(olm_tmp, data_value["olm"])
        theta_tmp = np.append(theta_tmp, data_value["theta_part"])
        
        
        if ( (idx+1)%10 == 0 ):
            
        
            data_mean_pyr.append( circmean(pyr_tmp, high=np.pi, low=-np.pi) )
            data_varience_pyr.append( circstd(pyr_tmp, high=np.pi, low=-np.pi) )
            
            data_mean_bas.append( circmean(bas_tmp, high=np.pi, low=-np.pi) )
            data_varience_bas.append( circstd(bas_tmp, high=np.pi, low=-np.pi) )
            
            data_mean_olm.append( circmean(olm_tmp, high=np.pi, low=-np.pi) )
            data_varience_olm.append( circstd(olm_tmp, high=np.pi, low=-np.pi) )
            
            theta_part.append(theta_tmp)
        
            pyr_tmp = np.array([], dtype=float)
            bas_tmp = np.array([], dtype=float)
            olm_tmp = np.array([], dtype=float)
            theta_tmp = np.array([], dtype=float)
    
    colors = get_colors()
    phases = np.linspace(-2*np.pi, 2*np.pi, 100)
    wave = np.median(number_synapses) + 20 * np.cos(phases)
    fig, axs = plt.subplots(nrows=1, ncols=3, sharey=True)
    
    axs[0].errorbar(data_mean_pyr, number_synapses, xerr=data_varience_pyr, fmt='o', color=colors["pyr"])
    axs[0].set_title('Pyr')
    
    axs[1].errorbar(data_mean_bas, number_synapses, xerr=data_varience_bas, fmt='o', color=colors["bas"])
    axs[1].set_title('Bas')
    
    axs[2].errorbar(data_mean_olm, number_synapses, xerr=data_varience_olm, fmt='o', color=colors["olm"])
    axs[2].set_title('OLM')
    
    for ax in axs:
        # ax.set_xlim(-np.pi, 3*np.pi)
        ax.plot(phases, wave, color="black", linewidth=2)
        ax.set_xlim(-2*np.pi, 2*np.pi)
    
    fig.tight_layout()
    fig.savefig(saving_path + ".png", dpi=500)
    
    
    theta_part_mean = np.zeros(number_synapses.size)
    theta_part_std = np.zeros(number_synapses.size)
    
    for idx, t in enumerate(theta_part):
        theta_part_mean[idx] = np.mean(t)
        theta_part_std[idx] = np.std(t)
        
        
    fig, axs = plt.subplots()
    
    axs.errorbar(number_synapses, theta_part_mean, yerr=theta_part_std, fmt='o')
    # axs.set_ylim(0, 0.3)
    axs.set_title('Relative theta power')
    
    fig.savefig(saving_path + "_theta_part.png", dpi=500)
    

def tonic_currents(processed_data, saving_path):
    tonic_carrents = np.linspace(-2, 1, 7)

    data_mean_pyr = []
    data_varience_pyr = []
    
    data_mean_bas = []
    data_varience_bas = []

    data_mean_olm = []
    data_varience_olm = []
    
    theta_part = []
    
    pyr_tmp = np.array([], dtype=float)
    bas_tmp = np.array([], dtype=float)
    olm_tmp = np.array([], dtype=float)
    theta_tmp = np.array([], dtype=float)
    
    for idx, data_value in enumerate(processed_data):
       
        
        pyr_tmp = np.append(pyr_tmp, data_value["pyramide"])
        bas_tmp = np.append(bas_tmp, data_value["basket"])
        olm_tmp = np.append(olm_tmp, data_value["olm"])
        theta_tmp = np.append(theta_tmp, data_value["theta_part"])
        
        
        if ( (idx+1)%10 == 0 ):
            
        
            data_mean_pyr.append( circmean(pyr_tmp, high=np.pi, low=-np.pi) )
            data_varience_pyr.append( circstd(pyr_tmp, high=np.pi, low=-np.pi) )
            
            data_mean_bas.append( circmean(bas_tmp, high=np.pi, low=-np.pi) )
            data_varience_bas.append( circstd(bas_tmp, high=np.pi, low=-np.pi) )
            
            data_mean_olm.append( circmean(olm_tmp, high=np.pi, low=-np.pi) )
            data_varience_olm.append( circstd(olm_tmp, high=np.pi, low=-np.pi) )
            
            theta_part.append(theta_tmp)
        
            pyr_tmp = np.array([], dtype=float)
            bas_tmp = np.array([], dtype=float)
            olm_tmp = np.array([], dtype=float)
            theta_tmp = np.array([], dtype=float)
    
    colors = get_colors()
    phases = np.linspace(-2*np.pi, 2*np.pi, 100)
    wave = np.median(tonic_carrents) + np.cos(phases)
    fig, axs = plt.subplots(nrows=1, ncols=3, sharey=True)
    
    axs[0].errorbar(data_mean_pyr, tonic_carrents, xerr=data_varience_pyr, fmt='o', color=colors["pyr"])
    axs[0].set_title('Pyr')
    
    axs[1].errorbar(data_mean_bas, tonic_carrents, xerr=data_varience_bas, fmt='o', color=colors["bas"])
    axs[1].set_title('Bas')
    
    axs[2].errorbar(data_mean_olm, tonic_carrents, xerr=data_varience_olm, fmt='o', color=colors["olm"])
    axs[2].set_title('OLM')
    
    for ax in axs:
        # ax.set_xlim(-np.pi, 3*np.pi)
        ax.plot(phases, wave, color="black", linewidth=2)
        ax.set_xlim(-2*np.pi, 2*np.pi)
    
    fig.tight_layout()
    fig.savefig(saving_path + ".png", dpi=500)
    
    
    theta_part_mean = np.zeros(tonic_carrents.size)
    theta_part_std = np.zeros(tonic_carrents.size)
    
    for idx, t in enumerate(theta_part):
        theta_part_mean[idx] = np.mean(t)
        theta_part_std[idx] = np.std(t)
        
        
    fig, axs = plt.subplots()
    
    axs.errorbar(tonic_carrents, theta_part_mean, yerr=theta_part_std, fmt='o')
    axs.set_ylim(0, 0.2)
    axs.set_title('Relative theta power')
    
    fig.savefig(saving_path + "_theta_part.png", dpi=500)    
    
    return None

####################################
def plot_whole_phase_distribution(processed_data, saving_path):

    
    pyr = np.array([], dtype=float)
    bas = np.array([], dtype=float)
    olm = np.array([], dtype=float)
       
    for idx, data_value in enumerate(processed_data):
       
        pyr = np.append(pyr, data_value["pyramide"])
        bas = np.append(bas, data_value["basket"])
        olm = np.append(olm, data_value["olm"])

    pyr = np.append(pyr, pyr + 2*np.pi )
    bas = np.append(bas, bas + 2*np.pi )
    olm = np.append(olm, olm + 2*np.pi )
    neurons_colors = get_colors()

    phases_x = np.linspace(-np.pi, 3*np.pi, 40)
    phases_y = 0.75*0.5*( np.cos(phases_x) + 1.0)
    plt.figure( tight_layout=True )
    plt.subplot(311)
    plt.hist(pyr, 40, normed=True, facecolor=neurons_colors["pyr"], alpha=0.75)
    plt.plot(phases_x, phases_y, color="black")
    plt.xlim(-np.pi, 3*np.pi)
    plt.ylim(0, 0.75)
    plt.subplot(312)
    plt.hist(bas, 40, normed=True, facecolor=neurons_colors["bas"], alpha=0.75)
    plt.plot(phases_x, phases_y, color="black")
    plt.xlim(-np.pi, 3*np.pi)
    plt.ylim(0, 0.75)
    plt.subplot(313)
    plt.hist(olm, 40, normed=True, facecolor=neurons_colors["olm"], alpha=0.75)
    plt.plot(phases_x, phases_y, color="black")
    plt.xlim(-np.pi, 3*np.pi)
    plt.ylim(0, 0.75)
    plt.tight_layout()
    plt.savefig(saving_path + "phase_disribution_histogram.png", dpi = 500)

def plot_theta_part(data, path):
    new_data = []
    
    
    
    for d in data:
        
        theta_part_tmp = np.array([], dtype=float)
        
        for t in d:
            theta_part_tmp = np.append(theta_part_tmp, t["theta_part"])
        
        new_data.append(theta_part_tmp)
    
    
    
    plt.figure()
    plt.boxplot(new_data, sym="")
    plt.tight_layout()
    plt.savefig(path + "theta_part.png", dpi = 500)
    
    
    
def keySorter(item):
    try:
        key = int(item[0:3])
    except ValueError:
        try:
            key = int(item[0:2])
        except ValueError:
            try:
                key = int(item[0])
            except ValueError:
                key = 0
    
    return key



septumInModel = False
main_path = "/home/ivan/Data/modeling_septo_hippocampal_model/hippocampal_model/" 

"""

path = main_path + "basic_model_test/"


control_data = []


for file in sorted( os.listdir(path), key = keySorter ):
    if os.path.splitext(file)[1] != ".npy":
        continue

    print (file)
    
    ret = make_figures(path, file, septumInModel)
    #ret = make_calculation(path, file)
    control_data.append( ret )
# plot_whole_phase_distribution(control_data, main_path)
"""




"""
path = main_path + "without_of_each_input/" # "basic_model_high_pp/"


high_pp = []


for file in sorted( os.listdir(path), key = keySorter ):
    if os.path.splitext(file)[1] != ".npy":
        continue
    
    
    tmp = file[0:2]
    try:
        tmp_int = int (tmp)
    except ValueError:
        try:
            tmp_int = int (tmp[0])
        except ValueError:
            continue
    
    if not(tmp_int >= 1 and tmp_int <= 10):
        continue
    
    
    print (file)
    
    ret = make_figures(path, file, septumInModel)
    #ret = make_calculation(path, file)
    high_pp.append( ret )

plot_whole_phase_distribution(high_pp, main_path + "(-ms1)_")


path = main_path + "without_of_each_input/"


low_cs = []


for file in sorted( os.listdir(path), key = keySorter ):
    if os.path.splitext(file)[1] != ".npy":
        continue
    tmp = file[0:2]
    try:
        tmp_int = int (tmp)
    except ValueError:
        continue
    # if (file != '51__all_results.npy'):continue
    if not(tmp_int >= 11 and tmp_int <= 20):
        continue




    print (file)
    
    ret = make_figures(path, file, septumInModel)
    #ret = make_calculation(path, file)
    low_cs.append( ret )
#
plot_whole_phase_distribution(low_cs, main_path + "(-ms2)_")
#data = [control_data, high_pp, low_cs]
#
#plot_theta_part(data, main_path)




path = main_path + "without_of_each_input/"


low_ms = []


for file in sorted( os.listdir(path), key = keySorter ):
    if os.path.splitext(file)[1] != ".npy":
        continue
    tmp = file[0:2]
    try:
        tmp_int = int (tmp)
    except ValueError:
        try:
            tmp_int = int (tmp[0])
        except ValueError:
            continue
    # if (file != '51__all_results.npy'):continue
    if not(tmp_int >= 41 and tmp_int <= 49):
        continue




    print (file)
    
    ret = make_figures(path, file, septumInModel)
    #ret = make_calculation(path, file)
    low_ms.append( ret )

plot_whole_phase_distribution(low_ms, main_path + "(-ms1and2)_")
data = [control_data, high_pp, low_cs, low_ms]

plot_theta_part(data, main_path)
"""



processed_data = []

# path = main_path + "ms_cs_pp_phase_shift2/"
# path = main_path + "without_of_each_input/"
# path = main_path + "no_rhythm_in_each_input(same frequency of random)/" # "no_rhythm_in_each_input/"
# path = main_path + "ms_pv1_pv2_phase_shift/"
# path = main_path + "bas_inputs_density/"
# path = main_path + "olm_inputs_density/"

# path = main_path + "bas_tonic_current/"
path = main_path + "olm_tonic_current/"
# path = main_path + "pyramide_dendrite_tonic_current/"
# path = main_path + "pyramide_soma_tonic_current/"
 
 
for file in sorted( os.listdir(path), key = keySorter ):
    if os.path.splitext(file)[1] != ".npy":
        continue
    # if (file != '33__all_results.npy'):continue
    print (file)
    
    # ret = make_figures(path, file, septumInModel)
    ret = make_calculation(path, file)
    processed_data.append( ret )
  


#ms_cs_pp_phase_shift_processing(processed_data, main_path)
#  without_of_each_input_processing(processed_data, control_data, main_path)
# without_of_each_input_processing(processed_data, control_data, main_path+"no_rhythms_same_frequency_")
# pv1_pv2_phase_shift(processed_data, main_path)
# balance_cs_bas2pyr(processed_data, main_path + "bas_inputs_density")
# balance_cs_bas2pyr(processed_data, main_path + "olm_inputs_density")

# tonic_currents(processed_data, main_path + "bas_tonic_current")
tonic_currents(processed_data, main_path + "olm_tonic_current")
# tonic_currents(processed_data, main_path + "pyr_dendrite_tonic_current")
# tonic_currents(processed_data, main_path + "pyr_soma_tonic_current")


