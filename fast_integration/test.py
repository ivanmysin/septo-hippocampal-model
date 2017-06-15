#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test script
"""

import fast_lib as lib
import numpy as np
import matplotlib.pyplot as plt
import time

neuron_params = {
    "V0" : -60.0,
    "C"  : 1.0,
    "n_currents" : 4,
}

external_current_params = {
    "Iext" : 0.5,
    "varience"  : 0.5,
    "type" : "external",     
}

leak_current_params = {
    "Erev" : -65.0,
    "gmax" : 0.1,
    "type" : "leak",
}

sodium_current4FS_params = {
    "Erev" : 50.0,
    "gmax" : 50.0,
    "fi"   : 10.0,
    "type" : "sodium_current4FS",
}

potassium_current4FS_params = {
    "Erev" : -90.0,
    "gmax" : 8.0,
    "fi"   : 10.0,
    "type" : "potassium_current4FS",
}

Neuron = lib.Compartment(neuron_params)

Neuron.addCurrent( lib.ExternalCurrent(external_current_params) )
Neuron.addCurrent( lib.LeakCurrent(leak_current_params) )
Neuron.addCurrent( lib.SodiumCurrent4FS(sodium_current4FS_params) )
Neuron.addCurrent( lib.PotassiumCurrent4FS(potassium_current4FS_params) )


dt = 0.1
duration=1000
t = time.time()
Neuron.run(dt=dt, duration=duration)
print (time.time() - t)

Vhist = Neuron.getVhist()
t = np.linspace(0, 0.001 * duration,  Vhist.size)

plt.plot(t, Vhist)
plt.xlim(0, 0.3)
plt.show()



