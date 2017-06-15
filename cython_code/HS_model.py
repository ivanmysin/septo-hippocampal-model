#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hippocampo-septal model
"""

# -*- coding: utf-8 -*-
"""
septo-hippocampal model
"""

import lib2 as lib
import numpy as np
#import matplotlib.pyplot as plt
import os
import time
import matplotlib.pyplot as plt
               

def run_model(path, params):

    cluster_pacemaker = {
        "V0": -50.0,
        "Iextmean": 1.2,        
        "Iextvarience": 0.5,
        "ENa": 50.0,
        "EK": -90.0,
        "El": -50.0,
        "Eh": -40.0,
        "gbarNa": 55.0,
        "gbarK": 8.0,
        "gbarKS": 12.0,
        "gbarH": 1.0,
        "gl": 0.1,
        "fi": 5,
    }
    
    fs_neuron = {
        "V0": -65.0,
        "Iextmean": 0.5,        
        "Iextvarience": 0.5,
        "ENa": 50.0,
        "EK": -90.0,
        "El": -65.0,
        "gbarNa": 55.0,
        "gbarK": 8.0,
        "gl": 0.1,   
        "fi": 10,
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
    
    HS_neuron_params = {
        "V0" : -63.0,
        "Iextmean" : 0.5,        
        "Iextvarience" : 0.0,
        "ENa" : 55.0,
        "EK"  : -90.0,
        "El"  : -65.0,
        "EH"  : -40.0,
        "ECa" : 120.0,
            
        "Ca"  : 0.0,
        "KD"  : 30.0,
             
        "gbarNa" : 35.0,
        "gbarK"  : 9.0,
        "gl"     : 0.1,   
        "gbarK_Ca" : 10,
        "gbarCa" : 1.0,
        "gbarH" : 0.15,    
        "alpha_Ca" : 0.002,
        "tau_Ca"   : 80,
    }
    
    ########## set model of septum ################
    # number of neurons
    Nglu = params["Nglu"]
    NgabaCR = params["NgabaCR"]
    NgabaPV1 = params["NgabaPV1"]
    NgabaPV2 = params["NgabaPV2"]
    NgabaPacPV1 = params["NgabaPacPV1"]
    NgabaPacPV2 = params["NgabaPacPV2"]
    NgabaHS = params["NgabaHS"]
    
    NN = Nglu + NgabaCR + NgabaPV1 + NgabaPV2 + NgabaPacPV1 + NgabaPacPV2
    
    Sparsness = params["sparsness"]  # sparsness in septal network
    
    indexes = {
        "glu" : [],
        "pv1" : [],
        "pv2" : [],
        "pv1_pac" : [],
        "pv2_pac" : [],
        "cr" : [],
        "hs" : [],
    }
    
    neurons = []
    for idx in range(Nglu):
        neuron = {
            "type" : "FS_Neuron",
            "compartments" : fs_neuron.copy()
        }
        neuron["compartments"]["V0"] += 10 * np.random.rand()
        neuron["compartments"]["Iextmean"] = 0.5 # 0.5*np.random.randn()
        
        indexes["glu"].append(len(neurons))
        neurons.append(neuron)
    
    for idx in range(NgabaCR):
        neuron = {
            "type" : "FS_Neuron",
            "compartments" : fs_neuron.copy()
        }
        neuron["compartments"]["V0"] += 10 * np.random.rand()
        neuron["compartments"]["Iextmean"] = 0 # 0.1*np.random.randn()
        indexes["cr"].append(len(neurons))
        neurons.append(neuron)
    
    for idx in range(NgabaPV1):
        neuron = {
            "type" : "FS_Neuron",
            "compartments" : fs_neuron.copy()
        }
        neuron["compartments"]["V0"] += 10 * np.random.rand()
        neuron["compartments"]["Iextmean"] = 0.1 #0.5*np.random.randn()
        indexes["pv1"].append(len(neurons))
        neurons.append(neuron)
    
    for idx in range(NgabaPacPV1):
        neuron = {
            "type" : "ClusterNeuron",
            "compartments" : cluster_pacemaker.copy()
        }
        neuron["compartments"]["V0"] += 10 * np.random.rand()
        neuron["compartments"]["Iextmean"] = 0.1 # 0.5*np.random.randn()
        indexes["pv1"].append(len(neurons))
        neurons.append(neuron)
    
    for idx in range(NgabaPV2):
        neuron = {
            "type" : "FS_Neuron",
            "compartments" : fs_neuron.copy()
        }
        neuron["compartments"]["V0"] += 10 * np.random.rand()
        neuron["compartments"]["Iextmean"] = 0.7 #0.5*np.random.randn()
        indexes["pv2"].append(len(neurons))
        neurons.append(neuron)
    
    for idx in range(NgabaPacPV2):
        neuron = {
            "type" : "ClusterNeuron",
            "compartments" : cluster_pacemaker.copy()
        }
        neuron["compartments"]["V0"] += 10 * np.random.rand()
        neuron["compartments"]["Iextmean"] = 0.7 #0.5*np.random.randn()
        indexes["pv2"].append(len(neurons))
        neurons.append(neuron)
    
    for idx in range(NgabaHS):
        neuron = {
            "type" : "HS_projective_neuron",
            "compartments" : HS_neuron_params.copy()
        }
        neuron["compartments"]["V0"] += 10 * np.random.rand()
        # neuron["compartments"]["Iextmean"] = 0.7 #0.5*np.random.randn()
        indexes["hs"].append(len(neurons))
        neurons.append(neuron)
    
    synapses = []
    # synapse from Glu to Glu neurons
    
    for idx1 in indexes["glu"]:
        for idx2 in indexes["glu"]:
            if (Sparsness < np.random.rand() or idx1 == idx2):
                continue
            pre_ind = idx1
            post_ind = idx2
            synapse = {
                "type" : "ComplexSynapse",
                "pre_ind": pre_ind, 
                "post_ind": post_ind,
                "pre_compartment_name": "soma",
                "post_compartment_name" : "soma",
                "params": ext_synapse_params.copy(),
            }
            synapse["params"]["w"] = 1
            synapses.append(synapse)
    
    # synapses from Glu naurons to GABA(CR) neurons
    for idx1 in indexes["glu"]:
        for idx2 in indexes["cr"]:
            if (Sparsness < np.random.rand() or idx1 == idx2):
                continue
            pre_ind = idx1
            post_ind = idx2
            synapse = {
                "type" : "ComplexSynapse",
                "pre_ind": pre_ind, 
                "post_ind": post_ind,
                "pre_compartment_name": "soma",
                "post_compartment_name" : "soma",
                "params": ext_synapse_params.copy(),
            }
            synapse["params"]["w"] = 0.2
            synapses.append(synapse)
    
    # synapses from GABA(CR) neurons to Glu naurons
    for idx1 in indexes["cr"]:
        for idx2 in indexes["glu"]:
            if (idx1 == idx2):
                continue
    
            pre_ind = idx1
            post_ind = idx2
            synapse = {
                "type" : "ComplexSynapse",
                "pre_ind": pre_ind, 
                "post_ind": post_ind,
                "pre_compartment_name": "soma",
                "post_compartment_name" : "soma",
                "params": inh_synapse_params.copy(),
            }
            synapse["params"]["w"] = 1.5
            synapses.append(synapse)
            
           
    # synapses from Glu neurons to PV1 and pac-PV1 naurons  
    for idx1 in indexes["glu"]:
        for idx2 in indexes["pv1"]:
            if (Sparsness < np.random.rand() or idx1 == idx2):
                continue
            pre_ind = idx1
            post_ind = idx2
            synapse = {
                "type" : "ComplexSynapse",
                "pre_ind": pre_ind, 
                "post_ind": post_ind,
                "pre_compartment_name": "soma",
                "post_compartment_name" : "soma",
                "params": ext_synapse_params.copy(),
            }
            synapse["params"]["w"] = 1
            synapses.append(synapse)
    
    # synapses from PV1 and pac-PV1 naurons to PV2 and pac-PV2
    for idx1 in indexes["pv1"]:
        for idx2 in indexes["pv2"]:
            if (Sparsness < np.random.rand() or idx1 == idx2):
                continue
            pre_ind = idx1
            post_ind = idx2
            synapse = {
                "type" : "ComplexSynapse",
                "pre_ind": pre_ind, 
                "post_ind": post_ind,
                "pre_compartment_name": "soma",
                "post_compartment_name" : "soma",
                "params": inh_synapse_params.copy(),
            }
            synapse["params"]["w"] = 0.5
            synapses.append(synapse) 
        
    # synapses from PV2 and pac-PV2 naurons to PV1 and pac-PV1  
    for pre_ind in indexes["pv2"]:
        for post_ind in indexes["pv1"]:
            if (Sparsness < np.random.rand() or idx1 == idx2):
                continue
            synapse = {
                "type" : "ComplexSynapse",
                "pre_ind": pre_ind, 
                "post_ind": post_ind,
                "pre_compartment_name": "soma",
                "post_compartment_name" : "soma",
                "params": inh_synapse_params.copy(),
            }
            synapse["params"]["w"] = 0.1
            synapses.append(synapse) 
    
    
    for s in synapses:
        s["params"]["w"] /= params["sparsness"]
        
          
    duration = params["duration"]
    dt = 0.1

    net = lib.Network(neurons, synapses)
    for _ in range(1):
        curent_time = time.time()
        net.integrate(dt, duration)
        print ("Calculation time is %0.3f sec" % (time.time() - curent_time) )
        net.save_results(path + "all_results")

#####################################################
Nglu = 40
NgabaCR = 40
NgabaPV1 = 40 
NgabaPV2 = 40
NgabaPacPV1 = 10
NgabaPacPV2 = 10
NgabaHS = 40


params = {
   "Nglu" : 40,        
   "NgabaCR" : 40,
   "NgabaPV1" : 40,  
   "NgabaPV2" : 40,
   "NgabaPacPV1" : 10,
   "NgabaPacPV2" : 10,
   "NgabaHS" : 40,
   "sparsness" : 0.5,
   "duration"  : 500,
}


path = "/home/ivan/Data/modeling_septo_hippocampal_model/HS_model/"
run_model(path, params)

file = path + "all_results.npy"
res = np.load(file)
firing = res[()]["results"]["firing"]

firing_slices = {}
# plt.figure( figsize = (10, 5), tight_layout=True   )
fig, (a1, a2, a3, a4, a5) = plt.subplots(5, 1, sharex=True, figsize = (10, 5)) #  gridspec_kw = {'height_ratios':[2, 2, 2, 1, 1, 8]},
fig.set_size_inches(10, 10)
   
firing[0, :] *= 0.001
# plot septal neurons
cum_it = Nglu
sl = firing[1, :] <= cum_it
firing_slices["glu"] = np.copy(sl)
glu_line = a1.scatter(firing[0, sl], firing[1, sl], color="r")
        
sl = (firing[1, :] > cum_it) & (firing[1, :] <= cum_it + NgabaCR)
firing_slices["gaba_cr"] = np.copy(sl)
cr_line = a2.scatter(firing[0, sl], firing[1, sl], color="g")
cum_it += NgabaCR
        
sl = (firing[1, :] > cum_it) & (firing[1, :] <= cum_it + NgabaPV1 + NgabaPacPV1)
firing_slices["gaba_pv1"] = np.copy(sl)
pv1_line = a3.scatter(firing[0, sl], firing[1, sl], color="b")
cum_it += NgabaPV1 + NgabaPacPV1
        
sl = (firing[1, :] > cum_it) & (firing[1, :] <= cum_it + NgabaPV2 + NgabaPacPV2)
firing_slices["gaba_pv2"] = np.copy(sl)
pv2_line = a4.scatter(firing[0, sl], firing[1, sl], color="b")
cum_it += NgabaPV2 + NgabaPacPV2

sl = (firing[1, :] > cum_it) & (firing[1, :] <= cum_it + params["NgabaHS"])
firing_slices["gaba_hs"] = np.copy(sl)
hs_line = a5.scatter(firing[0, sl], firing[1, sl], color="m")  
    
fig.tight_layout()
# fig.savefig(saving_path + "raster.png", dpi = 500)