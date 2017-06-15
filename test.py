# -*- coding: utf-8 -*-
"""
test
"""
import numpy as np
import matplotlib.pyplot as plt
import time

class FS_neuron:
    
    def __init__(self, params):
         self.V = params["V0"]
         self.Iextmean = params["Iextmean"]        
         self.Iextvarience = params["Iextvarience"]
         self.ENa = params["ENa"]
         self.EK = params["EK"]
         self.El = params["El"]
         self.gbarNa = params["gbarNa"]
         self.gbarK = params["gbarK"]
         self.gl = params["gl"]   
         self.fi = params["fi"]
         
         self.Iext = np.random.normal(self.Iextmean, self.Iextvarience) 
         
         self.Vhist = []
         self.firing = []
         
         self.m = self.alpha_m() / (self.alpha_m() + self.beta_m())
         self.n = self.alpha_n() / (self.alpha_n() + self.beta_n())
         self.h = self.alpha_h() / (self.alpha_h() + self.beta_h())

         self.gNa = self.gbarNa * self.m * self.m * self.m * self.h
         self.gK = self.gbarK * self.n * self.n * self.n * self.n

         self.Isyn = 0
         self.countSp = True
         self.th = -20


    def alpha_m(self):
            #double alpha;
         x = -0.1 * (self.V + 33)
         if (x == 0):
            x = 0.000000001
         alpha = x / ( np.exp(x) - 1 )
         return alpha
#########
    def beta_m(self):
        beta = 4 * np.exp(- (self.V + 58) / 18 )
        return beta


########
    def alpha_h(self):

        alpha = self.fi * 0.07 * np.exp( -(self.V + 51) / 10)
        return alpha

########
    def beta_h(self):

        beta = self.fi / ( np.exp(-0.1 * (self.V + 21)) + 1 )
        return beta
    
########
    def alpha_n(self):
    
        x = -0.1 * (self.V + 38)
        if ( x==0 ):
            x = 0.00000000001
        
        alpha = self.fi * 0.1 * x / (np.exp(x) - 1)
        return alpha
#######np.

    def beta_n(self):
    
        return (self.fi * 0.125 * np.exp( -(self.V + 48 )/ 80))

#######
    def h_integrate(self, dt):
    
        h_0 = self.alpha_h() / (self.alpha_h() + self.beta_h())
        tau_h = 1 / (self.alpha_h() + self.beta_h())
        return h_0 -(h_0 - self.h) * np.exp(-dt/tau_h)
#######

    def n_integrate(self, dt):

        n_0 = self.alpha_n() / (self.alpha_n() + self.beta_n() )
        tau_n = 1 / (self.alpha_n() + self.beta_n())
        return n_0 -(n_0 - self.n) * np.exp(-dt/tau_n)

#######
    def integrate (self,  dt, duraction):

        t = 0
        i = 0
        while (t < duraction):
            self.Vhist.append(self.V)
            
            self.V = self.V + dt * (self.gNa * (self.ENa - self.V) + self.gK * (self.EK - self.V) + self.gl*(self.El - self.V) - self.Isyn + self.Iext)
    
            self.m = self.alpha_m() / (self.alpha_m() + self.beta_m())
            self.n = self.n_integrate(dt)
            self.h = self.h_integrate(dt)
            
            self.gNa = self.gbarNa * self.m * self.m * self.m * self.h
            self.gK = self.gbarK * self.n * self.n * self.n * self.n
    
            self.Iext = np.random.normal(self.Iextmean, self.Iextvarience) 
    
            self.Isyn = 0
            i += 1
            t += dt
  

########
    def checkFired(self, t_):
    
        if (self.V >= self.th and self.countSp):
            self.firing.append(t_)
            self.countSp = False
        
        if (self.V < self.th):
            self.countSp = True
######## 
    def getV(self):
        return self.V
        
    def getVhist(self):
        return self.Vhist
        
    def setIsyn(self, Isyn):
        self.Isyn += Isyn
    
    def setIext(self, Iext):
        self.Iext = Iext
    
    def getFiring(self):
        return self.firing
###################################################################################################################


class FS_neuron_fast:
    
    def __init__(self, params):
         self.V = params["V0"]
         self.Iextmean = params["Iextmean"]        
         self.Iextvarience = params["Iextvarience"]
         self.ENa = params["ENa"]
         self.EK = params["EK"]
         self.El = params["El"]
         self.gbarNa = params["gbarNa"]
         self.gbarK = params["gbarK"]
         self.gl = params["gl"]   
         self.fi = params["fi"]
         
         self.Iext = np.random.normal(self.Iextmean, self.Iextvarience) 
         
         self.Vhist = []
         self.firing = []
         
         self.npoits4dump = 1000
         self.make_dumps()
         
         self.m = self.alpha_m(self.V) / (self.alpha_m(self.V) + self.beta_m(self.V))
         self.n = self.alpha_n(self.V) / (self.alpha_n(self.V) + self.beta_n(self.V))
         self.h = self.alpha_h(self.V) / (self.alpha_h(self.V) + self.beta_h(self.V))

         self.gNa = self.gbarNa * self.m * self.m * self.m * self.h
         self.gK = self.gbarK * self.n * self.n * self.n * self.n

         self.Isyn = 0
         self.countSp = True
         self.th = -20
         
         
         
         
    
    def make_dumps(self):
        V = np.linspace(-100, 50, self.npoits4dump) + 0.000000001
        self.Vstep4dumping = 150 / self.npoits4dump
        
        self.alpha_m_dump = self.alpha_m(V)
        self.beta_m_dump = self.beta_m(V)
        self.alpha_h_dump = self.alpha_h(V)
        self.beta_h_dump = self.beta_h(V)
        self.alpha_n_dump = self.alpha_n(V)
        self.beta_n_dump = self.beta_n(V)
        

    def alpha_m(self, V, dump=False):
        
        if (dump):
             idx = int( (V + 100) / self.Vstep4dumping )
             return self.alpha_m_dump[idx]
        else:
             x = -0.1 * (V + 33)
             # if (x == 0):
             #    x = 0.000000001
             alpha = x / ( np.exp(x) - 1 )
             return alpha
     
        
#########
    def beta_m(self, V, dump=False):
        if (dump):
            idx = int( (V + 100) / self.Vstep4dumping )
            return self.beta_m_dump[idx]
        else:
            beta = 4 * np.exp(- (V + 58) / 18 )
            return beta


########
    def alpha_h(self, V, dump=False):
        if (dump):
            idx = int( (V + 100) / self.Vstep4dumping )
            return self.alpha_h_dump[idx]
        else:
            alpha = self.fi * 0.07 * np.exp( -(V + 51) / 10)
            return alpha

########
    def beta_h(self, V, dump=False):
        if (dump):
            idx = int( (V + 100) / self.Vstep4dumping )
            return self.beta_h_dump[idx]
        else:
            beta = self.fi / ( np.exp(-0.1 * (V + 21)) + 1 )
            return beta

    
########
    def alpha_n(self, V, dump=False):
        if (dump):
            idx = int( (V + 100) / self.Vstep4dumping )
            return self.alpha_n_dump[idx]
        else:
        
            x = -0.1 * (V + 38)
            # if ( x==0 ):
            #     x = 0.00000000001
            
            alpha = self.fi * 0.1 * x / (np.exp(x) - 1)
            return alpha

#######np.

    def beta_n(self, V, dump=False):
        if (dump):
            idx = int( (V + 100) / self.Vstep4dumping )
            return self.beta_n_dump[idx]
        else:
            return (self.fi * 0.125 * np.exp( -(V + 48 )/ 80))


#######
    def h_integrate(self, dt):
    
        h_0 = self.alpha_h(self.V, True) / (self.alpha_h(self.V, True) + self.beta_h(self.V, True))
        tau_h = 1 / (self.alpha_h(self.V, True) + self.beta_h(self.V, True))
        return h_0 -(h_0 - self.h) * np.exp(-dt/tau_h)
#######

    def n_integrate(self, dt):

        n_0 = self.alpha_n(self.V, True) / (self.alpha_n(self.V, True) + self.beta_n(self.V, True) )
        tau_n = 1 / (self.alpha_n(self.V, True) + self.beta_n(self.V, True))
        return n_0 -(n_0 - self.n) * np.exp(-dt/tau_n)

#######
    def integrate (self,  dt, duraction):

        t = 0
        i = 0
        while (t < duraction):
            self.Vhist.append(self.V)
            
            self.V = self.V + dt * (self.gNa * (self.ENa - self.V) + self.gK * (self.EK - self.V) + self.gl*(self.El - self.V) - self.Isyn + self.Iext)
    
            self.m = self.alpha_m(self.V, True) / (self.alpha_m(self.V, True) + self.beta_m(self.V, True))
            self.n = self.n_integrate(dt)
            self.h = self.h_integrate(dt)
            
            self.gNa = self.gbarNa * self.m * self.m * self.m * self.h
            self.gK = self.gbarK * self.n * self.n * self.n * self.n
    
            self.Iext = np.random.normal(self.Iextmean, self.Iextvarience) 
    
            self.Isyn = 0
            i += 1
            t += dt
  

########
    def checkFired(self, t_):
    
        if (self.V >= self.th and self.countSp):
            self.firing.append(t_)
            self.countSp = False
        
        if (self.V < self.th):
            self.countSp = True
######## 
    def getV(self):
        return self.V
        
    def getVhist(self):
        return self.Vhist
        
    def setIsyn(self, Isyn):
        self.Isyn += Isyn
    
    def setIext(self, Iext):
        self.Iext = Iext
    
    def getFiring(self):
        return self.firing


















params = {
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


n = FS_neuron_fast(params)

ctime = time.time()
n.integrate(0.1, 10000)
print ("Integrated time is %d sec" % (time.time()-ctime) )
V = n.getVhist()

plt.plot(V)

