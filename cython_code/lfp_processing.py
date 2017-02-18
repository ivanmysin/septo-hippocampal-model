# -*- coding: utf-8 -*-
"""
main script for processing
"""

import numpy as np
import matplotlib.pyplot as plt
import processingLib as plib
import os
import scipy.signal as sig
def make_figures(path, file, septumInModel=True):
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


   
    saving_path = path + file[0:2]
    
    file = path + file
    
    fd = 10000
    margin = 0.5 # margin from start for analisis in seconds
    
    res = np.load(file)
    
    lfp = res[()]["results"]["lfp"]
    margin_ind = int(margin*fd)
    lfp = lfp[margin_ind : ]
    V = res[()]["results"]["V"]
    currents = res[()]["results"]["currents"]
    firing = res[()]["results"]["firing"]
    firing[0, :] -= 1000*margin
    firing = firing[:, firing[0, :] >= 0]
    
    
    
    lfp_filtred = plib.filtrate_lfp(lfp, fd)
    lfp_filtred = lfp_filtred[0:-1:20]
    fd = 500
    t = np.linspace(0, lfp_filtred.size/fd, lfp_filtred.size)
    
    
    lfp2 = 0
    for v in V:
        try:
            lfp2 += v["dendrite"] - v["soma"]
        except KeyError:
            continue
    lfp2 = lfp2[margin_ind : ]
    lfp_filtred2 = plib.filtrate_lfp(lfp2, 10000)
    lfp_filtred2 = lfp_filtred2[0:-1:20]
    plt.figure()
    plt.subplot(211)
    plt.plot(t, lfp_filtred2, "b")
    plt.subplot(212)
    plt.plot(t, lfp_filtred, "g")
    plt.savefig(saving_path + "lfp2.png", dpi = 500)
    
    lfp = lfp2 # !!!!!!!!!!
    lfp_filtred = lfp_filtred2 # !!!!!!!!!!!!!!!!
    
    # calculate and plot wavelet spectrum and LFP signal
    freqs, coefs = plib.computemycwt(fd, lfp_filtred)
    
    plt.figure()
    plt.subplot(211)
    Z = np.abs(coefs)
    plt.pcolor(t, freqs, Z, cmap='rainbow', vmin=Z.min(), vmax=Z.max())
    plt.title('Wavelet spectrum of simulated LFP')
    # set the limits of the plot to the limits of the data
    plt.axis([t[0], t[-1], freqs[0], freqs[-1]])
    plt.colorbar()
    
    
    plt.subplot(212)
    plt.plot(t, lfp_filtred, color="black", linewidth=2)
    plt.xlim(t[0], t[-1])
    #plt.ylim(1.2*lfp_filtred.max(), -1.2*lfp_filtred.min())
    plt.colorbar()
    plt.savefig(saving_path + "wavelet.png", dpi = 500)
    
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
 
    
    # calculate and plot phase coupling between units activity and theta rhythm 
    theta_lfp = plib.butter_bandpass_filter(lfp_filtred, 4, 10, fd, 3)
    plt.figure()
    plt.plot(t, theta_lfp)
    plt.title("Signal in Theta diapason")    
    # plot raster of neuronal activity
    
    firing_slices = {}
    plt.figure()
    plt.subplot(311)
    plt.plot(t, lfp_filtred, color="black", linewidth=2)
    plt.xlim(0, 1.5) # plt.xlim(t[0], t[-1])
    
    plt.subplot(312)
    plt.plot(t, theta_lfp, color="blue", linewidth=2)
    plt.xlim(0, 1.5) #plt.xlim(t[0], t[-1])
    
    plt.subplot(313)
    firing[0, :] *= 0.001
    # plot septal neurons
    if (septumInModel):
        cum_it = Nglu
        sl = firing[1, :] <= cum_it
        firing_slices["glu"] = np.copy(sl)
        glu_line = plt.scatter(firing[0, sl], firing[1, sl], color="r", s=1)
        
        sl = (firing[1, :] > cum_it) & (firing[1, :] <= cum_it + NgabaCR)
        firing_slices["gaba_cr"] = np.copy(sl)
        cr_line = plt.scatter(firing[0, sl], firing[1, sl], color="g", s=1)
        cum_it += NgabaCR
        
        sl = (firing[1, :] > cum_it) & (firing[1, :] <= cum_it + NgabaPV1 + NgabaPacPV1)
        firing_slices["gaba_pv1"] = np.copy(sl)
        pv1_line = plt.scatter(firing[0, sl], firing[1, sl], color="b", s=1)
        cum_it += NgabaPV1 + NgabaPacPV1
        
        sl = (firing[1, :] > cum_it) & (firing[1, :] <= cum_it + NgabaPV2 + NgabaPacPV2)
        firing_slices["gaba_pv2"] = np.copy(sl)
        pv2_line = plt.scatter(firing[0, sl], firing[1, sl], color="b", s=1)
        cum_it += NgabaPV2 + NgabaPacPV2

        # plot hippocampal neurons
        sl = (firing[1, :] > cum_it) & (firing[1, :] <= cum_it + Np)
    else:
        cum_it = Np 
        sl = (firing[1, :] < cum_it)
    firing_slices["pyramide"] = np.copy(sl)
    pyr_line = plt.scatter(firing[0, sl], firing[1, sl], color="r", s=1)
    
    if (septumInModel):
        cum_it += Np
    
        
    sl = (firing[1, :] > cum_it) & (firing[1, :] <= cum_it + Nb)
    firing_slices["basket"] = np.copy(sl)
    basket_line = plt.scatter(firing[0, sl], firing[1, sl], color="k", s=1)
    cum_it += Nb
    sl = (firing[1, :] > cum_it) & (firing[1, :] <= cum_it + Nolm)
    firing_slices["olm"] = np.copy(sl)
    olm_line = plt.scatter(firing[0, sl], firing[1, sl], color="m", s=1)
    
    sl = (firing[1, :] > cum_it) & (firing[1, :] <= cum_it + Nolm)
    firing_slices["olm"] = np.copy(sl)
    
    if not(septumInModel):
        cum_it += Nolm
        sl = (firing[1, :] > cum_it) & (firing[1, :] <= cum_it + NSG)
                
        pv_line = plt.scatter(firing[0, sl], firing[1, sl], color="b", s=1)
        firing_slices["gaba_pv1"] = np.copy(sl)
    """
    if (septumInModel):          
        plt.legend((glu_line, cr_line, pv1_line, pv2_line, pyr_line, basket_line, olm_line),
               ('Glu', 'GABA(CR)', 'GABA(PV1)', 'GABA(PV2)', 'Pyramide', 'Basket', 'OLM'),
               scatterpoints=1,
               loc='upper left',
               ncol=1,
               fontsize=12)
    else:
        plt.legend((pv_line, pyr_line, basket_line, olm_line),
               ('GABA(PV)',  'Pyramide', 'Basket', 'OLM'),
               scatterpoints=1,
               loc='upper left',
               ncol=1,
               fontsize=12)
    """ 
    cum_it += NSG 
    sl = (firing[1, :] > cum_it)    
    cs_line = plt.scatter(firing[0, sl], firing[1, sl], color="g", s=1)
    firing_slices["cs"] = np.copy(sl)
               
    plt.xlim(0, 1.5) #(0, t[-1])
    plt.ylim(0, 800)
    plt.tight_layout()
    plt.savefig(saving_path + "raster.png", dpi = 500)
    
  
    
    
    neurons_phases = plib.get_units_disrtibution(theta_lfp, fd, firing, firing_slices)
    for neuron_phase_key in neurons_phases.keys():
        neurons_phases[neuron_phase_key] = np.append(neurons_phases[neuron_phase_key], neurons_phases[neuron_phase_key] + 2*np.pi )
    phases_x = np.linspace(-np.pi, 3*np.pi, 40)
    phases_y = 0.5*np.cos(phases_x) + 0.5
    plt.figure()
    plt.subplot(311)
    plt.hist(neurons_phases["pyramide"], 40, normed=True, facecolor='red', alpha=0.75)
    plt.plot(phases_x, phases_y, c="b")
    plt.xlim(-np.pi, 3*np.pi)
    plt.subplot(312)
    plt.hist(neurons_phases["basket"], 40, normed=True, facecolor='black', alpha=0.75)
    plt.plot(phases_x, phases_y, c="b")
    plt.xlim(-np.pi, 3*np.pi)
    plt.subplot(313)
    plt.hist(neurons_phases["olm"], 40, normed=True, facecolor='m', alpha=0.75)
    plt.plot(phases_x, phases_y, c="b")
    plt.xlim(-np.pi, 3*np.pi)
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
        if (key == "gaba_pv1" or key == "gaba_pv2"):
            color = "b"
        if (key == "pyramide"):
            color = "r"
        if (key == "basket"):
            color = "k"  
        if (key == "olm"):
            color = "m" 
        if (key == "cs"):
            color = "g"
        plt.scatter(angles, length, color=color, s=2)
        
        # calculate histogram ?
    """
    plt.legend((glu_line, cr_line, pv1_line, pv2_line, pyr_line, basket_line, olm_line),
               ('Glu', 'GABA(CR)', 'GABA(PV1)', 'GABA(PV2)', 'Pyramide', 'Basket', 'OLM'),
               scatterpoints=1,
               loc='upper left',
               ncol=1,
               fontsize=12)
    """
    
    plt.savefig(saving_path + "phase_disribution_of_neurons.png", dpi = 500)
    plt.figure()
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
        if (key == "gaba_pv1" or key == "gaba_pv2"):
            color = "b"
        if (key == "pyramide"):
            color = "r"
        if (key == "basket"):
            color = "k"  
        if (key == "olm"):
            color = "m" 
        if (key == "cs"):
            color = "g"
        plt.scatter(angles, numbers, color=color, s=2)

        # calculate histogram ?
    plt.ylim(0, 800)
    tmp_phases = np.linspace(-np.pi, np.pi, 1000)
    plt.plot( tmp_phases, 200 * ( np.cos(tmp_phases) + 1 ), "-b" )
    plt.xlim(-np.pi, np.pi)
    
    plt.savefig(saving_path + "phase_disribution_of_neurons2.png", dpi = 500)
    """
    plt.legend((glu_line, cr_line, pv1_line, pv2_line, pyr_line, basket_line, olm_line),
               ('Glu', 'GABA(CR)', 'GABA(PV1)', 'GABA(PV2)', 'Pyramide', 'Basket', 'OLM'),
               scatterpoints=1,
               loc='upper left',
               ncol=1,
               fontsize=12)
    """
    
    
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
    plt.figure()
    plt.subplot(411)
    plt.plot(t, Vin)
    plt.title("Intracellular potential on soma")
    plt.xlim(1, 1.5) #plt.xlim(1, 1.5)
    plt.subplot(412)
    plt.plot(t, soma_currents, "b")
    plt.plot(t, lfp_filtred_normed, "g")
    plt.ylim(-1, 1)
    plt.xlim(1, 1.5) # plt.xlim(1, 1.5) #
    plt.title("soma")
    plt.subplot(413)
    plt.plot(t, dendrite_current, "b")
    plt.plot(t, lfp_filtred_normed, "g")
    plt.xlim(1, 1.5) # plt.xlim(1, 1.5) #
    # plt.ylim(-1, 1)
    plt.title("dendrite")
    plt.subplot(414)    
    # plot septal neurons
    if (septumInModel):
        cum_it = Nglu
        sl = firing[1, :] <= cum_it
        firing_slices["glu"] = np.copy(sl)
        glu_line = plt.scatter(firing[0, sl], firing[1, sl], color="r", s=1)
        
        sl = (firing[1, :] > cum_it) & (firing[1, :] <= cum_it + NgabaCR)
        firing_slices["gaba_cr"] = np.copy(sl)
        cr_line = plt.scatter(firing[0, sl], firing[1, sl], color="g", s=1)
        cum_it += NgabaCR
        
        sl = (firing[1, :] > cum_it) & (firing[1, :] <= cum_it + NgabaPV1 + NgabaPacPV1)
        firing_slices["gaba_pv1"] = np.copy(sl)
        pv1_line = plt.scatter(firing[0, sl], firing[1, sl], color="b", s=1)
        cum_it += NgabaPV1 + NgabaPacPV1
        
        sl = (firing[1, :] > cum_it) & (firing[1, :] <= cum_it + NgabaPV2 + NgabaPacPV2)
        firing_slices["gaba_pv2"] = np.copy(sl)
        pv2_line = plt.scatter(firing[0, sl], firing[1, sl], color="b", s=1)
        cum_it += NgabaPV2 + NgabaPacPV2

        # plot hippocampal neurons
        sl = (firing[1, :] > cum_it) & (firing[1, :] <= cum_it + Np)
    else:
        cum_it = Np 
        sl = (firing[1, :] < cum_it)
    firing_slices["pyramide"] = np.copy(sl)
    pyr_line = plt.scatter(firing[0, sl], firing[1, sl], color="c", s=1)
    
    if (septumInModel):
        cum_it += Np
    
        
    sl = (firing[1, :] > cum_it) & (firing[1, :] <= cum_it + Nb)
    firing_slices["basket"] = np.copy(sl)
    basket_line = plt.scatter(firing[0, sl], firing[1, sl], color="k", s=1)
    cum_it += Nb
    sl = (firing[1, :] > cum_it) & (firing[1, :] <= cum_it + Nolm)
    firing_slices["olm"] = np.copy(sl)
    olm_line = plt.scatter(firing[0, sl], firing[1, sl], color="m", s=1)
    plt.ylim(0, 800)
    #plt.xlim(1, 1.5) #plt.xlim(0, 1.5)
    if not(septumInModel):
        cum_it += Nolm
        sl = (firing[1, :] > cum_it)
                
        pv_line = plt.scatter(firing[0, sl], firing[1, sl], color="b", s=1)
        firing_slices["gaba_pv1"] = np.copy(sl)
    
    if (septumInModel):          
        plt.legend((glu_line, cr_line, pv1_line, pv2_line, pyr_line, basket_line, olm_line),
               ('Glu', 'GABA(CR)', 'GABA(PV1)', 'GABA(PV2)', 'Pyramide', 'Basket', 'OLM'),
               scatterpoints=1,
               loc='upper left',
               ncol=1,
               fontsize=8)
    else:
        plt.legend((pv_line, pyr_line, basket_line, olm_line),
               ('GABA(PV)',  'Pyramide', 'Basket', 'OLM'),
               scatterpoints=1,
               loc='upper left',
               ncol=1,
               fontsize=8)
    
    
    
    plt.savefig(saving_path + "currents.png", dpi = 500)
    
    
     
####################################
"""
path = "/home/ivan/Data/modeling_septo_hippocampal_model/septo_hippocampal_model2/without_hipp-septum_input/"
path = "/home/ivan/Data/modeling_septo_hippocampal_model/septo_hippocampal_model2/OLM2PV1/"
path = "/home/ivan/Data/modeling_septo_hippocampal_model/septo_hippocampal_model2/OLM2PV1andPV2/"
path = "/home/ivan/Data/modeling_septo_hippocampal_model/septo_hippocampal_model2/OLM2PV2/"

path = "/home/ivan/Data/modeling_septo_hippocampal_model/hippocampal_model/only_one_rhytm/"
path = "/home/ivan/Data/modeling_septo_hippocampal_model/hippocampal_model/different_phase_shift/"
for file in sorted(os.listdir(path)):
    if (file[0] == "." or file[-4:] != ".npy"):
        continue
    make_figures(path, file, False)
    print (file)
"""
"""
# for seizures analisis
path = "/home/ivan/Data/modeling_septo_hippocampal_model/seisures/norm/"
file = "_all_results.npy"
make_figures(path, file, False)
"""



path = "/home/ivan/Data/modeling_septo_hippocampal_model/hippocampal_model/test/"
file = path + "1__all_results.npy"
ret = make_figures(path, "1__all_results.npy", False)




"""
firing[0, :] *= 0.001
basket_cells = firing # firing[:, (firing[1, :] > 0)&(firing[1, :] < 400)]




i_soma = -currents[0]["soma"]
i_dendrite = -currents[0]["dendrite"] - 1
fd = 10000
t = np.linspace(0, i_soma.size/fd, i_soma.size)
"""




##plt.subplot(211)
#plt.plot(t, i_soma, c="r")
##plt.subplot(212)
#plt.plot(t, i_dendrite, c="b")
#
#i_soma_fft =  2 * np.abs(np.fft.rfft(i_soma)) / i_soma.size
#i_dendrite_fft =  2 * np.abs(np.fft.rfft(i_dendrite)) / i_dendrite.size
#frqs = np.fft.rfftfreq(i_soma.size, d=1./fd)
#
#plt.figure()
#plt.subplot(211)
#plt.plot(frqs, i_soma_fft, c="r")
#plt.xlim(1, 30)
#plt.subplot(212)
#plt.plot(frqs, i_dendrite_fft, c="b")
#plt.xlim(1, 30)
"""
sum_i = lfp # i_soma #+ i_dendrite
frqs = np.fft.rfftfreq(sum_i.size, d=1./fd)

sum_i_fft = np.fft.rfft(sum_i)
sum_i_fft[frqs > 100] = 0
sum_i = np.fft.irfft(sum_i_fft)

sum_i_fft_teta = np.copy(sum_i_fft)
sum_i_fft_teta[frqs > 10] = 0
sum_i_teta = np.fft.irfft(sum_i_fft_teta)


sum_i_fft_teta2 = np.copy(sum_i_fft)
sum_i_fft_teta2[(frqs < 10)&(frqs < 15)] = 0
sum_i_teta2 = np.fft.irfft(sum_i_fft_teta2)

sum_i_fft =  2 * np.abs(sum_i_fft) / sum_i.size

s = V[0]["soma"]-sum_i
plt.figure()
plt.subplot(211)
plt.plot(t, sum_i, c="b")
#plt.plot(t, sum_i_teta, c="r")
#plt.plot(t, sum_i_teta2, c="g")
plt.xlim(t[0], t[-1])
plt.subplot(212)


plt.scatter(basket_cells[0,:], basket_cells[1,:])
plt.xlim(t[0], t[-1])
plt.ylim(0, 700)


plt.figure()
plt.plot(frqs, sum_i_fft)
plt.xlim(1, 30)
"""