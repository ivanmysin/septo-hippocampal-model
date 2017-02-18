# -*- coding: utf-8 -*-
"""
model of seizures in hippocampus
"""

import lib2 as lib
import numpy as np
import os
#import time


class SimmulationParams:
    def __init__(self, params=None, mode="default"):
        self.p = params
        self.mode = mode
      
    def iext_function(self, neuron_ind, compartment_name, t):
        return 0
          
        t = 0.001 * t
        Iext = 0
        if (neuron_ind >= 100 and neuron_ind < 105):
            Iext = np.cos(2 * np.pi * t * 8) + 1

        if (neuron_ind >= 105):
            Iext = np.cos(2 * np.pi * t * 8 + 2.65) + 1
        return Iext
    def set_params(self, params):
        self.p = params
        
    
    def set_mode(self, mode):
        self.mode = mode
        
    def get_mode(self):
        return self.mode
#########################################################
saving_fig_path = "/home/ivan/Data/modeling_septo_hippocampal_model/seisures/"
sim = SimmulationParams()
sim.set_mode("only_one_rhytm")
sim.set_params([1, 1])

path = saving_fig_path + "norm/"
if not( os.path.isdir(path) ):
    os.mkdir(path)        
        
        
#########################################################
if __name__ == "__main__":
    soma_params = {
            "V0": 0.0,
            "C" : 3.0,
            "Iextmean": 0.5,        
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
            "Iextmean": 0.5,        
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
        "alpha_s" : 1.1,
        "beta_s": 0.19,
        "K": 5.0,
        "teta": 2.0,
        "w" : 1.0,
        "delay" : 0,
    }
    
    inh_synapse_params = {
        "Erev" : -15.0,
        "gbarS": 0.005,
        "alpha_s" : 14.0,
        "beta_s": 0.07,
        "K": 2.0,
        "teta": 0.0,
        "w" : 1.0,
        "delay" : 0,
    }
    
    if (sim.get_mode() == "variate_frequency"):
        CosSpikeGeneratorParams["freq"] = sim.p
        
   
    ##########################
    neurons = []
    synapses = []
    Np = 400 # number of pyramide neurons
    Nb = 50   # number of basket cells
    Nolm = 50 # number of olm cells
    Ns = 1600 # number synapses between pyramide cells
    Nint2pyr = 100 # 4 # number synapses from one interneuron to one pyramide neuron 
    NSG = 0 # number of spike generators
    Ns2in = 400 # number synapses from septal generators to hippocampal interneurons 
    
    
    for idx in range(Np):
        soma = {"soma" : soma_params.copy()}
        soma["soma"]["V0"] = 0.5 * np.random.randn()
        soma["soma"]["Iextmean"] += 0.5*np.random.randn()
        
        dendrite = {"dendrite" : dendrite_params.copy()}
        dendrite["dendrite"]["V0"] = soma["soma"]["V0"]
        #dendrite["dendrite"]["Iextmean"] += 0.5*np.random.randn()
       
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
         #neuron["compartments"]["Iextmean"] += 0.5*np.random.randn()
         neurons.append(neuron)
         
    for idx in range(Nolm):
        neuron = {
            "type" : "olm_cell",
            "compartments" : olm_params.copy()
        }
        neuron["compartments"]["V0"] += 10 * np.random.rand()
        #neuron["compartments"]["Iextmean"] += 0.5*np.random.randn()
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
           "type" : "ComplexSynapse",
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
               "type" : "ComplexSynapse",
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
               "type" : "ComplexSynapse",
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
           "type" : "ComplexSynapse",
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
           "type" : "ComplexSynapse",
           "pre_ind": pre_ind, 
           "post_ind": post_ind,
           "pre_compartment_name": "soma",
           "post_compartment_name" : "soma",
           "params": ext_synapse_params.copy(),
           
        }
        synapse["params"]["delay"] = np.random.randint(20, 50)
        synapses.append(synapse)
    
    
    
    """
    for idx in range(Ns2in):
        pre_ind = np.random.randint(Np + Nb + Nolm, Np + Nb + Nolm + NSG//2)
        post_ind = np.random.randint(Np, Np + Nb)
        synapse = {
            "type" : "ComplexSynapse",
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
            "type" : "ComplexSynapse",
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
        if (syn["params"]["Erev"] == 60):
            syn["params"]["w"] *= 2
            
        if (syn["params"]["Erev"] == -15):
            syn["params"]["w"] /= 2

    dt = 0.1
    duration = 500
    net = lib.Network(neurons, synapses)
    
    for _ in range(4):
        
        net.integrate(dt, duration, sim.iext_function)
        net.save_results(path + "_all_results")