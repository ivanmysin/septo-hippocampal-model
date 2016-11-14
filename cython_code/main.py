# -*- coding: utf-8 -*-
"""
main script
"""

import lib2 as lib
import numpy as np
#from  scipy.signal import medfilt
import processingLib as plib
import matplotlib.pyplot as plt
import os
import time

class SimmulationParams:
    def __init__(self, params=None, mode="default"):
        self.p = params
        self.mode = mode
      
    # variate frequency from septum 
    def iext_function(self, neuron_ind, compartment_name, t):
        return 0
          
        t = 0.001 * t
        Iext = 0
        if (neuron_ind >= 100 and neuron_ind < 105):
            Iext = np.cos(2 * np.pi * t * 8) + 1

        if (neuron_ind >= 105):
            Iext = np.cos(2 * np.pi * t * 8 + 2.65) + 1
        return Iext
               
        
        
        """        
        if self.mode == "default":
            return 0
        
        if self.mode == "variate_frequency":

            if compartment_name == "soma":
                Iext = np.cos(2 * np.pi * t * self.p) - 1
           
            if compartment_name == "dendrite":
                Iext = np.cos(2 * np.pi * t * self.p + 2.65) - 1
            # Iext *= 2
            return Iext
        
        # one rhytm
        if self.mode == "only_one_rhytm":
  
            if compartment_name == "soma":
                Iext = self.p[0] * np.cos(2 * np.pi * t * 8) - 1
                
            if compartment_name == "dendrite":
                Iext = self.p[1] * np.cos(2 * np.pi * t * 8 + 2.65) - 1
            
            if Iext == 0:
                Iext = -np.random.rand()
            Iext *= 0.5
            return Iext
        
        if self.mode == "different_phase_shift":
            
            if compartment_name == "soma":
                Iext = np.cos(2 * np.pi * t * 6) - 1
                
            if compartment_name == "dendrite":
                Iext = np.cos(2 * np.pi * t * 6 + self.p) - 1
            #Iext *= 2
            return Iext
            
        return 0
    """
    
    def set_params(self, params):
        self.p = params
        
    
    def set_mode(self, mode):
        self.mode = mode
        
    def get_mode(self):
        return self.mode

def run_model(sim, path):
    soma_params = {
            "V0": 0.0,
            "C" : 3.0,
            "Iextmean": -0.5,        
            "Iextvarience": 0.2,
            "ENa": 120.0,
            "EK": -15.0,
            "El": 0.0,
            "ECa": 140.0,
            "CCa": 0.05,
            "sfica": 0.13,
            "sbetaca": 0.075, 
            "gbarNa": 30.0,
            "gbarK_DR": 17.0,
            "gbarK_AHP": 0.8,        
            "gbarK_C " : 15.0,
            "gl": 0.1,
            "gbarCa": 6.0,
        
    }
    
    dendrite_params = {
            "V0": 0.0,
            "C" : 3.0,
            "Iextmean": -0.5,        
            "Iextvarience": 0.2,
            "ENa": 120.0,
            "EK": -15.0,
            "El": 0.0,
            "ECa": 140.0,
            "CCa": 0.05,
            "sfica": 0.13, 
            "sbetaca": 0.075, 
            "gbarNa": 0.0,
            "gbarK_DR": 0.0,
            "gbarK_AHP": 0.8,        
            "gbarK_C " : 5.0,
            "gl": 0.1,
            "gbarCa": 5.0,
    }
    
    connection_params = {
        "compartment1": "soma",
        "compartment2": "dendrite",
        "p": 0.5,
        "g": 1.5,
    }
    
    basket_fs_neuron = {
         "V0": -65.0,
         "Iextmean": 0.2,        
         "Iextvarience": 0.2,
         "ENa": 50.0,
         "EK": -90.0,
         "El": -65.0,
         "gbarNa": 55.0,
         "gbarK": 8.0,
         "gl": 0.1,   
         "fi": 10,
    }
    
    olm_params = {
        "V0" : -70,
        "gl" : 0.05,
        "El" : -70,
        "gbarNa" : 10.7, 
        "ENa" : 90,
        "gbarK" : 31.9,
        "EK" : -100,
        "gbarKa" : 16.5,
        "gbarH" : 0.05,
        "EH" : -32.9,
        "Iextmean" : 0.2,        
        "Iextvarience": 0.2,
    }
    
    CosSpikeGeneratorParams = {
       "freq"  : 6.0, # frequency in Hz
       "phase" : 0.0, # phase in rad
       "latency" :  10.0, # in ms
       "probability" : 0.01,
    }
    
    PoisonSpikeGenerator = {
       "latency" :  10.0, # in ms
       "probability" : 0.00001,
    }
    
    ext_synapse_params = {
        "Erev" : 60.0,
        "gbarS": 0.005,
        "tau" : 2.0,
        "w" : 5.0,
        "delay" : 50,
    }

    inh_synapse_params = {
        "Erev" : -15.0,
        "gbarS": 0.005,
        "tau" : 5.0,
        "w" : 10.0,
        "delay" : 50,
    }
    
    if (sim.get_mode() == "variate_frequency"):
        CosSpikeGeneratorParams["freq"] = sim.p
        
   
    ##########################
    neurons = []
    synapses = []
    Np = 400 # number of pyramide neurons
    Nb = 50   # number of basket cells
    Nolm = 50 # number of olm cells
    Ns = 400 # number synapses between pyramide cells
    Nint2pyr = 50 # 4 # number synapses from one interneuron to one pyramide neuron 
    NSG = 20 # number of spike generators
    Ns2in = 400 # number synapses from septal generators to hippocampal interneurons 
    
    
    for idx in range(Np):
        soma = {"soma" : soma_params.copy()}
        soma["soma"]["V0"] = 0.5 * np.random.randn()
        soma["soma"]["Iextmean"] += 0.5*np.random.randn()
        
        dendrite = {"dendrite" : dendrite_params.copy()}
        dendrite["dendrite"]["V0"] = soma["soma"]["V0"]
        dendrite["dendrite"]["Iextmean"] += 0.5*np.random.randn()
       
        connection = connection_params.copy()
        neuron = {
            "type" : "pyramide",
            "compartments" : [soma, dendrite],
            "connections" : [connection]
        }
        neurons.append(neuron)
    
    for idx in range(Nb):
         neuron = {
            "type" : "FS_Neuron",
            "compartments" : basket_fs_neuron.copy()
         }
         neuron["compartments"]["V0"] += 10 * np.random.rand()
         neuron["compartments"]["Iextmean"] += 0.5*np.random.randn()
         neurons.append(neuron)
         
    for idx in range(Nolm):
        neuron = {
            "type" : "olm_cell",
            "compartments" : olm_params.copy()
        }
        neuron["compartments"]["V0"] += 10 * np.random.rand()
        neuron["compartments"]["Iextmean"] += 0.5*np.random.randn()
        neurons.append(neuron)
    
    for idx in range(NSG):
        
        if (sim.get_mode() == "only_one_rhytm" and sim.p[1] == 0 and idx >= NSG//2):
            
            neuron = {
                "type" : "PoisonSpikeGenerator", 
                "compartments" : PoisonSpikeGenerator.copy()
            }
            neurons.append(neuron)
            continue  
        
        if (sim.get_mode() == "only_one_rhytm" and sim.p[0] == 0 and idx < NSG//2):
            
            neuron = {
                "type" : "PoisonSpikeGenerator", 
                "compartments" : PoisonSpikeGenerator.copy()
            }
            neurons.append(neuron)
            continue
        
        
        neuron = {
            "type" : "CosSpikeGenerator", 
            "compartments" : CosSpikeGeneratorParams.copy()
        }
        if (idx >= NSG//2):
            if (sim.get_mode() == "different_phase_shift"):
                neuron["compartments"]["phase"] = sim.p
            else:
                neuron["compartments"]["phase"] = 2.65 # !!!!
        
        neurons.append(neuron)
    
    for idx in range(Ns):
        pre_ind = np.random.randint(0, Np)
        post_ind = np.random.randint(0, Np)
        if (pre_ind == post_ind):
            post_ind = np.random.randint(0, Np)
        synapse = {
           "type" : "SimpleSynapseWithDelay",
           "pre_ind": pre_ind, 
           "post_ind": post_ind,
           "pre_compartment_name": "soma",
           "post_compartment_name" : "soma",
           "params" : ext_synapse_params.copy(),
        }
        synapse["params"]["delay"] = np.random.randint(20, 50)
        synapses.append(synapse)
    
    for _ in range(Nint2pyr):
        for idx in range(Np):
            pre_ind = np.random.randint(Np, Np + Nb)
            
            synapse = {
               "type" : "SimpleSynapseWithDelay",
               "pre_ind": pre_ind, 
               "post_ind": idx,
               "pre_compartment_name": "soma",
               "post_compartment_name" : "soma",
               "params": inh_synapse_params.copy(),
            }
            synapse["params"]["delay"] = np.random.randint(20, 50)
            synapses.append(synapse)
    
    for _ in range(Nint2pyr):
        for idx in range(Np):
            pre_ind = np.random.randint(Np + Nb, Np + Nb + Nolm)
            synapse = {
               "type" : "SimpleSynapseWithDelay",
               "pre_ind": pre_ind, 
               "post_ind": idx,
               "pre_compartment_name": "soma",
               "post_compartment_name" : "dendrite",
               "params": inh_synapse_params.copy(),
            }
            synapse["params"]["delay"] = np.random.randint(20, 50)
            synapses.append(synapse)
    
    
    
      
    for idx in range(Np):
        pre_ind = idx
        
        # set synapse from pyramide to OLM neuron
        post_ind = np.random.randint(Np + Nb, Np + Nb + Nolm)
        synapse = {
           "type" : "SimpleSynapseWithDelay",
           "pre_ind": pre_ind, 
           "post_ind": post_ind,
           "pre_compartment_name": "soma",
           "post_compartment_name" : "soma",
           "params": ext_synapse_params.copy(),
           
        }
        synapse["params"]["delay"] = np.random.randint(20, 50)
        synapses.append(synapse)
        
        # set synapse from pyramide to basket neuron
        post_ind = np.random.randint(Np, Np + Nb)
        synapse = {
           "type" : "SimpleSynapseWithDelay",
           "pre_ind": pre_ind, 
           "post_ind": post_ind,
           "pre_compartment_name": "soma",
           "post_compartment_name" : "soma",
           "params": ext_synapse_params.copy(),
           
        }
        synapse["params"]["delay"] = np.random.randint(20, 50)
        synapses.append(synapse)
    
    
    
    
    for idx in range(Ns2in):
        pre_ind = np.random.randint(Np + Nb + Nolm, Np + Nb + Nolm + NSG//2)
        post_ind = np.random.randint(Np, Np + Nb)
        synapse = {
            "type" : "SimpleSynapseWithDelay",
            "pre_ind": pre_ind, 
            "post_ind": post_ind,
            "pre_compartment_name": "soma",
            "post_compartment_name" : "soma",
            "params": inh_synapse_params.copy(),
            }
        # synapse["params"]["Erev"] = -75
        synapse["params"]["w"] = 100
        synapse["params"]["delay"] = np.random.randint(20, 50)
        synapses.append(synapse)
    
    
    for idx in range(Ns2in):
        pre_ind = np.random.randint(Np + Nb + Nolm + NSG//2, Np + Nb + Nolm + NSG)
        post_ind = np.random.randint(Np + Nb, Np + Nb + Nolm)
        synapse = {
            "type" : "SimpleSynapseWithDelay",
            "pre_ind": pre_ind, 
            "post_ind": post_ind,
            "pre_compartment_name": "soma",
            "post_compartment_name" : "soma",
            "params": inh_synapse_params.copy()
        }
        # synapse["params"]["Erev"] = -75
        synapse["params"]["w"] = 100
        synapse["params"]["delay"] = np.random.randint(20, 50)
        synapses.append(synapse)
    
    
    """
    for syn in synapses:
        print (neurons[syn["pre_ind"]]["type"] + " -> " + neurons[syn["post_ind"]]["type"])
        print ( str(syn["pre_ind"]) + " -> " + str(syn["post_ind"]) )
    return 
    """
    
    dt = 0.1
    duration = 500
    net = lib.Network(neurons, synapses)
    
    for _ in range(4):
        
        net.integrate(dt, duration, sim.iext_function)
        
        V = net.getVhist()
        VmeanSoma = 0
        VmeanDendrite = 0
        np.save(path + "V", V)
        for v in V:
            try:
                VmeanSoma += v["soma"]
                VmeanDendrite += v["dendrite"]
            except KeyError:
                continue
            
        VmeanSoma /= Np
        VmeanDendrite /= Np
        t = np.linspace(duration-500, duration, VmeanSoma.size)
        plt.figure()
        plt.subplot(211)
        plt.plot(t, VmeanSoma, "b")
        plt.subplot(212)
        plt.plot(t, VmeanDendrite, "r")
        
        plt.savefig(path + "mean_V", dpi=300)
        lfp = net.getLFP()
    
        lfp = plib.butter_bandpass_filter(lfp, 1, 80, 1000/dt, 3)
        np.save(path + "lfp", lfp)
        
        currents = net.getfullLFP()
        np.save(path + "  currents", currents)
       
        plt.figure()
        plt.plot(t, lfp)
        #plt.xlim(1000, 1500)
        # lfp_half = lfp[t > duration/2]
        plt.savefig(path + "lfp", dpi=300)
        
        fft_y = np.abs(np.fft.rfft(lfp))/lfp.size
        fft_x = np.fft.rfftfreq(lfp.size, 0.001*dt)
        plt.figure()
        plt.plot(fft_x[1:], fft_y[1:])
        plt.xlim(2, 50)
        
        theta_power = np.sum(fft_y[(fft_x>4)&(fft_x<12)])/np.sum(fft_y)
        plt.savefig(path + "spectra_of_lfp", dpi=300)
        firing = net.getFiring()
        np.save(path + "firing", firing)
        
        
        plt.figure()
        
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
        plt.savefig(path + "raster", dpi=300)
        plt.show()
        
        duration += 500
        print (theta_power)
    return theta_power
saving_fig_path = "/home/ivan/Data/modeling_septo_hippocampal_model/"
sim = SimmulationParams()
#############################
"""
# variate frequency from septum
theta_power = np.zeros([20, 5], dtype=float)

path = saving_fig_path + "variate_frequency/"
if not( os.path.isdir(path) ):
    os.mkdir(path)

p = np.linspace(1, 20, 20)
sim.set_mode("variate_frequency")
idx2 = -1
idx3 = 0
for idx in range(100):
    if (idx%5 == 0):
        idx2 += 1
        sim.set_params(p[idx2]) 
        
        idx3 = 0
    path_tmp = path + str(idx + 1) + "_"
    theta = run_model(sim, path_tmp)
    theta_power[idx2, idx3] = theta
    idx3 += 1

plt.figure()
plt.boxplot(theta_power.T)
plt.ylabel("theta power on soma")
plt.xlabel("frequency of septum output")
plt.savefig(saving_fig_path + "variate_septum_frequency.png")
plt.show()

###########################
"""

"""
# one rhytm
theta_power = np.zeros([15], dtype=float)

sim.set_mode("only_one_rhytm")
sim.set_params([1, 1])

path = saving_fig_path + "only_one_rhytm/"
if not( os.path.isdir(path) ):
    os.mkdir(path)
    
for idx in range(15):
    if (idx == 5):
        sim.set_params([1, 0])

    if (idx == 10):
        sim.set_params([0, 1])
    
    path_tmp = path + str(idx + 1) + "_"
    theta = run_model(sim, path_tmp)
    theta_power[idx] = theta
    

    

plt.figure()
plt.boxplot( [theta_power[0:10], theta_power[10:20], theta_power[20:30]] )
plt.xticks([1, 2, 3], ["control", "-dendrite", "-soma"])
plt.ylabel("theta power on soma")
plt.savefig(saving_fig_path + "one_rhythm.png")
plt.show()
"""



"""
# phase difference research
sim.set_mode("different_phase_shift")
sim.set_params([0, 1])

theta_power = np.zeros([2, 5], dtype=float)
p = np.array([2.65, np.pi], dtype=float)   #np.linspace(-np.pi, np.pi, 20)

path = saving_fig_path + "different_phase_shift/"
if not( os.path.isdir(path) ):
    os.mkdir(path)

idx2 = 0
idx3 = 0
for idx in range(10):
    

    if (idx%5 == 0):
        sim.set_params(p[idx2])
        idx2 += 1
        idx3 = 0

    path_tmp = path + str(idx + 1) + "_"
    

    theta = run_model(sim, path_tmp)

    theta_power[idx2, idx3] = theta
    idx3 += 1
    
    
    

plt.figure()
plt.errorbar(p, np.mean(theta_power, axis=1), yerr=np.std(theta_power, axis=1), fmt='o')
plt.ylabel("theta power on soma")
plt.xlabel("phase shift, rad")
plt.savefig(saving_fig_path + "phase_shift.png")
plt.show()
"""


theta_power = np.zeros([3], dtype=float)

sim.set_mode("only_one_rhytm")
sim.set_params([1, 1])

path = saving_fig_path + "test/"
if not( os.path.isdir(path) ):
    os.mkdir(path)
    
for idx in range(3):
    path_tmp = path + str(idx + 1) + "_"
    t = time.time()
    theta = run_model(sim, path_tmp)
    print (time.time() - t)
    theta_power[idx] = theta

#lib.testqueue()


    