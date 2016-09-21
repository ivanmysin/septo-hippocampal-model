# -*- coding: utf-8 -*-
"""
CA1 neuron from
    A Two Compartment Model of a CA1 Pyramidal Neuron
        Katie A. Ferguson∗†and Sue Ann Campbell
        (2009)
"""
#from math import exp
import numpy as np

exp = np.exp
 
class Compartment:
    def __init__(self, params):
        self.V = params["V0"]
        self.Capacity = params["C"]
        
        self.Iextmean = params["Iextmean"]        
        self.Iextvarience = params["Iextvarience"]
        
        self.ENa = params["ENa"]
        self.EK = params["EK"]
        self.El = params["El"]
        self.ECa = params["ECa"]
        
        self.CCa = params["CCa"]
        self.sfica = params["sfica"]
        self.sbetaca = params["sbetaca"]
        
        self.gbarNa = params["gbarNa"]
        self.gbarK_DR = params["gbarK_DR"]

        self.gbarK_AHP = params["gbarK_AHP"]        
        self.gbarK_C = params["gbarK_C "]        

        self.gl = params["gl"]
        self.gbarCa = params["gbarCa"]
        
        self.Vhist = []
        self.firing = []
        self.th = -20
        
        self.m = self.alpha_m() / (self.alpha_m() + self.beta_m())
        self.h = self.alpha_h() / (self.alpha_h() + self.beta_h())
        self.n = self.alpha_n() / (self.alpha_n() + self.beta_n())
        self.s = self.alpha_s() / (self.alpha_s() + self.beta_s())
        self.c = self.alpha_c() / (self.alpha_c() + self.beta_c())
        self.q = self.alpha_q() / (self.alpha_q() + self.beta_q())
        
        self.calculate_currents()
        
    def calculate_currents(self):
        self.Il = self.gl * (self.V - self.El)
        self.INa = self.gbarNa * self.m * self.m * self.h * (self.V - self.ENa)
        self.IK_DR = self.gbarK_DR * self.n * (self.V - self.EK)
        self.IK_AHP = self.gbarK_AHP * self.q * (self.V - self.EK)
        self.IK_C = self.gbarK_C * self.c * (self.V - self.EK)
        
        
        tmp = self.CCa / 250.0
        if (tmp < 1):
            self.IK_C *= tmp    
        
        self.ICa = self.gbarCa * self.s * self.s * (self.V - self.ECa)
        self.Iext = np.random.normal(self.Iextmean, self.Iextvarience)
                                                        
        self.Isyn = 0
        
        
    def alpha_m(self):
        x = 13.1 - self.V
        if (x == 0):
            x = 0.000001
        alpha = 0.32 * x / (exp(0.25 * x) - 1)
        return alpha
        
        
    def beta_m(self):
        x = self.V - 40.1
        if (x == 0):
            x = 0.00001
        beta = 0.28 * x / (exp(0.2 * x) - 1)
        return beta
        
    def alpha_h(self):
        alpha = 0.128 * exp((17 - self.V) / 18)
        return alpha
        
    def beta_h(self):
        x = 40 - self.V 
        if (x == 0):
            x = 0.00001
        beta = 4 / (exp(0.2 * x) + 1)
        return beta

    def alpha_n(self):
        x = 35.1 - self.V
        if (x == 0):
            x = 0.00001
        alpha = 0.016 * x / (exp(0.2 * x) - 1)
        return alpha

    def beta_n(self):
        beta = 0.25 * exp(0.5 - 0.025 * self.V)
        return beta
        
    def alpha_s(self):
        x = self.V - 65
        alpha = 1.6 / (1 + exp(-0.072 * x))
        return alpha
    
    def beta_s(self):
        x = self.V - 51.1
        if (x == 0):
            x = 0.00001
        beta = 0.02 * x / (exp(0.2 * x) - 1)
        return beta

    def alpha_c(self):
        if(self.V > 50):
            alpha = 2 * exp((6.5 - self.V)/27)
        else:
            alpha = exp( ((self.V - 10)/11) - ((self.V - 6.5)/27) ) / 18.975   
        return alpha
    
    def beta_c(self):
        if (self.V > 0):
            beta = 0
        else:
            beta = 2 * exp((6.5 - self.V)/27) - self.alpha_c()
        return beta
    
    def alpha_q(self):
        alpha = 0.00002 * self.CCa
        if (alpha > 0.01):
            alpha = 0.01
        return alpha
    
    def beta_q(self):
        return 0.001
    

    def h_integrate(self, dt):
        h_0 = self.alpha_h() / (self.alpha_h() + self.beta_h())
        tau_h = 1 / (self.alpha_h() + self.beta_h())
        return h_0 - (h_0 - self.h) * exp(-dt / tau_h)


    def n_integrate(self, dt):
        n_0 = self.alpha_n() / (self.alpha_n() + self.beta_n() )
        tau_n = 1 / (self.alpha_n() + self.beta_n())
        return n_0 - (n_0 - self.n) * exp(-dt / tau_n)
        
    def s_integrate(self, dt):
        s_0 = self.alpha_s() / (self.alpha_s() + self.beta_s() )
        tau_s = 1 / (self.alpha_s() + self.beta_s())
        return s_0 - (s_0 - self.s) * exp(-dt / tau_s)
    
    def c_integrate(self, dt):
        c_0 = self.alpha_c() / (self.alpha_c() + self.beta_c() )
        tau_c = 1 / (self.alpha_c() + self.beta_c())
        return c_0 - (c_0 - self.c) * exp(-dt / tau_c)
    
    def q_integrate(self, dt):
        q_0 = self.alpha_q() / (self.alpha_q() + self.beta_q() )
        tau_q = 1 / (self.alpha_q() + self.beta_q())
        return q_0 - (q_0 - self.q) * exp(-dt / tau_q)
    
    def CCa_integrate(self, dt):
        k1 = self.CCa
        k2 = k1 + 0.5 * dt * (- self.sfica * self.ICa - self.sbetaca * k1)
        k3 = k2 + 0.5 * dt * (- self.sfica * self.ICa - self.sbetaca * k2)
        k4 = k1 + dt * (- self.sfica * self.ICa - self.sbetaca * k1)        
        return (k1 + 2*k2 + 2*k3 + k4) / 6
        
        
        
    def integrate(self, dt, duration):
        t = 0
        self.Vhist.append(self.V)
        while (t < duration):
            self.V += dt * (-self.Il - self.INa - self.IK_DR - self.IK_AHP - self.IK_C - self.ICa - self.Isyn + self.Iext) / self.Capacity
            
            self.m = self.alpha_m() / (self.alpha_m() + self.beta_m())
            self.h = self.h_integrate(dt)
            self.n = self.n_integrate(dt)
            self.s = self.s_integrate(dt)
            self.c = self.c_integrate(dt)
            self.q = self.q_integrate(dt)
            
            self.CCa = self.CCa_integrate(dt)
       
       
       
       
            self.calculate_currents()
             
            t += dt
    
    
    def checkFired(self, t_):
    
        if (self.V >= self.th and self.countSp):
            self.firing.append(t_)
            self.countSp = False
        
        if (self.V < self.th):
            self.countSp = True

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

class IntercompartmentConnection:
    def __init__(self, soma, dendrite, g, p):
        self.soma = soma
        self.dendrite = dendrite
        self.g = g
        self.p = p
    
    def integrate(self):
        
        Isoma = -(self.g / self.p) * (self.dendrite.getV() - self.soma.getV())
        Idendite = -(self.g/(1 - self.p)) * (self.soma.getV() - self.dendrite.getV())
        
        self.soma.setIsyn(Isoma)
        self.dendrite.setIsyn(Idendite)
        
class ComplexNeuron:
    def __init__(self, compartments, connections):
        self.soma = compartments[0]
        self.compartments = compartments
        self.connections = connections
        self.getV = self.soma.getV 
    
    def integrate(self, dt, duration):
        t = 0
        while(t < duration):
            for p in self.compartments:
                p.integrate(dt, dt)
                
            for c in self.connections:
                c.integrate()
            
            t += dt
    def getSoma(self):
        return self.soma
    
    def getDendrite(self):
        return self.compartments[1]

class Synapse:
            
    def __init__ (self, pre, post, params):
    
        self.pre = pre
        self.post = post
        self.Erev = params["Erev"]
        self.gbarS = params["gbarS"]
        self.tau = params["tau"]
        self.w = params["w"]
        
        self.S = 0
        self.Isyn = 0


    def integrate(self, dt):
    
        Vpre = self.pre.getV() # V of pre neuron
        
        if (Vpre > 40):
            self.S = 1
        
        if ( self.S < 0.005 ):
            self.S = 0
            return
    
        Vpost = self.post.getV()
        self.Isyn = self.w * self.gbarS * self.S * (Vpost - self.Erev)
        self.post.setIsyn(self.Isyn) #  Isyn for post neuron
        
        k1 = self.S
        k2 = k1 - 0.5 * dt * (self.tau * k1)
        k3 = k2 - 0.5 * dt * (self.tau * k2)
        k4 = k1 - dt * (self.tau * k1)
        
        self.S = (k1 + 2*k2 + 2*k3 + k4) / 6.0 
        
class OLMcell:
    def __init__(self, params):
        self.V = params["V0"]
        self.Capacity = params["C"]
        
        self.Iextmean = params["Iextmean"]        
        self.Iextvarience = params["Iextvarience"]
        
        self.ENa = params["ENa"]
        self.EK = params["EK"]
        self.El = params["El"]
        self.EH = params["EH"]
        
        self.gbarNa = params["gbarNa"]
        self.gbarNap = params["gbarNap"]
        self.gbarK_DR = params["gbarK_DR"]
        self.gbarH = params["gbarH"]
       
        self.gl = params["gl"]
       
        
        self.Vhist = []
        self.firing = []
        self.th = -20
        
        self.m = self.alpha_m() / (self.alpha_m() + self.beta_m())
        self.mpo = self.alpha_mpo() / (self.alpha_mpo() + self.beta_mpo())
        self.h = self.alpha_h() / (self.alpha_h() + self.beta_h())
        self.n = self.alpha_n() / (self.alpha_n() + self.beta_n())
        
        self.lambda_fo_H = self.lambda_finf_H()
        self.lambda_so_H = self.lambda_sinf_H()    
        
        self.calculate_currents()
        
        
    def calculate_currents(self):
        self.Il = self.gl * (self.V - self.El)
        self.INa = (self.gbarNa * self.m * self.m * self.m * self.h + self.gbarNap * self.mpo) * (self.V - self.ENa)
        
        self.IK_DR = self.gbarK_DR * self.n * self.n * self.n * self.n * (self.V - self.EK)
        self.IH = self.gbarH * (0.65 * self.lambda_fo_H + 0.35 * self.lambda_so_H ) * (self.V - self.EH)
        self.Iext = np.random.normal(self.Iextmean, self.Iextvarience)                                                  
        self.Isyn = 0
        
        
    def alpha_m(self):
        x = self.V + 54
        if x==0: x = 0.000001
        alpha = 0.32 * x / (1 - exp(-0.25 * x) )
        return alpha
        
        
    def beta_m(self):
        x = self.V + 27
        if x==0: x = 0.000001
        beta = 0.28 * x / (exp(0.2 * x) - 1)
        return beta
        
    def alpha_h(self):
        alpha = 0.128 * exp( -(self.V + 50) / 18 )
        return alpha
        
    def beta_h(self):
        x = self.V + 27
        if x==0: x = 0.000001
        beta = 4 / (1 + exp(-0.2 * x) )
        return beta
    
    def alpha_mpo(self):
        x = self.V + 38
        if x == 0: x = 0.000001
        alpha = 1 / (0.15 * ( 1 + exp(-6.5*x) ) )
        return alpha

    def beta_mpo(self):
        x = self.V + 38
        if x == 0: x=0000000.1
        beta = exp(-6.5*x) / ( 0.15 * (1 + exp(-6.5*x)) )
        return beta        
        
    def alpha_n(self):
        x = self.V + 52
        if x == 0: x = 0.00001
        alpha = 0.032 * x / (1 - exp(-0.2*x))        
        return alpha

    def beta_n(self):
        beta = 0.5 * exp( -(self.V + 57) / 40 )
        return beta
        
    def lambda_finf_H(self):
        x = self.V + 79.2
        if x == 0: x = 0.00001
        lambda_inf = 1 / (1 + 9.78*x)
        return lambda_inf
        
    def tau_finf_H(self):
        tau = 0.51 / ( exp((self.V - 1.7)/10) + exp(-(self.V + 340)/52 ) ) + 1
        return tau
    
    def lambda_sinf_H(self):
        x = self.V + 2.83
        if x == 0: x = 0.00001
        lambda_inf = 1 / (1 + exp(x/15.9) )**58
        return lambda_inf
    
    def tau_sinf_H(self):
        tau = 5.6 / ( exp( (self.V - 1.7) / 10) + exp( -(self.V + 340) / 52 ) )  + 1
        return tau

    def h_integrate(self, dt):
        h_0 = self.alpha_h() / (self.alpha_h() + self.beta_h())
        tau_h = 1 / (self.alpha_h() + self.beta_h())
        return h_0 - (h_0 - self.h) * exp(-dt / tau_h)


    def n_integrate(self, dt):
        n_0 = self.alpha_n() / (self.alpha_n() + self.beta_n() )
        tau_n = 1 / (self.alpha_n() + self.beta_n())
        return n_0 - (n_0 - self.n) * exp(-dt / tau_n)
        
  
    def mpo_integrate(self, dt):
        mpo_0 = self.alpha_mpo() / (self.alpha_mpo() + self.beta_mpo() )
        tau_mpo = 1 / (self.alpha_mpo() + self.beta_mpo())
        return mpo_0 - (mpo_0 - self.mpo) * exp(-dt / tau_mpo)
        
    def lambda_fo_H_integrate(self, dt):
        lambda_foinf = self.lambda_finf_H()
        tau_inf =  self.tau_finf_H()
        return lambda_foinf - (lambda_foinf - self.lambda_fo_H) * exp(-dt / tau_inf)
        
    def lambda_so_H_integrate(self, dt):
        lambda_so_inf = self.lambda_sinf_H()
        tau_inf = self.tau_sinf_H()
        return lambda_so_inf - (lambda_so_inf - self.lambda_so_H) * exp(-dt / tau_inf)
        
    
     
    def integrate(self, dt, duration):
        t = 0
        
        while (t < duration):
            self.Vhist.append(self.V)

            # self.Iext = np.cos(2*np.pi*t*6)            
            
            self.V += dt * (-self.Il - self.INa - self.IK_DR - self.IH - self.Isyn + self.Iext) / self.Capacity
            
            self.m = self.alpha_m() / (self.alpha_m() + self.beta_m())
            self.mpo = self.mpo_integrate(dt)
            self.h = self.h_integrate(dt)
            self.n = self.n_integrate(dt)
            self.lambda_so_H = self.lambda_so_H_integrate(dt) 
            self.lambda_fo_H = self.lambda_fo_H_integrate(dt)
           
            self.calculate_currents()
             
            t += dt
    
    
    def checkFired(self, t_):
    
        if (self.V >= self.th and self.countSp):
            self.firing.append(t_)
            self.countSp = False
        
        if (self.V < self.th):
            self.countSp = True

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
        