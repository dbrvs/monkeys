
# coding: utf-8

# # Nested models
# 
# In this script, I have all the models that I use in the nested model analysis. These go from most complicated and most free parameters, down to least complicated and least free parameters. Each heading describes the feature that was removed to make the model less complicated.
# 
# The cheat sheet is as follows:
# 
# 1. complete
# 2. same natural death rate U and P
# 3. E doesn't get recruited by presence of U cells
# 4. no U
# 5. no E
# 6. no E and no U
# 

import numpy as np
from scipy.integrate import odeint

#initial conditions that are always true regardless of model
U0=0      #no infected cells
P0=0      #productively infected cells
V0=0.03   #start with 30 copies per mL

# # complete model
# (13 free parameters + 5 initial conditions)
# - susceptible, infected cells, and virus
# - productive and unproductively infected cells (different natural death rates)
# - adaptive compartment removes infecteds (both U and P)
# - adaptive compartment recruits more adaptive cells based on number of infected (both U and P)
# - adaptive saturates at some E50
def model_1(X,t,aS,dS,Bt,tau,dU,dP,k,aE,dE,E50,w,p,g):  
    dY = np.zeros(5);
    S=X[0]; U=X[1]; P=X[2]; E=X[3]; V=X[4];    
    dY[0] = aS - dS*S - Bt*S*V               #susceptible cells
    dY[1] = (1-tau)*Bt*S*V - dU*U - k*E*U    #unproductively infected
    dY[2] = tau*Bt*S*V - dP*P - k*E*P        #productively infected
    dY[3] = w*(P+U)*E/(E+E50) + aE - dE*E;   #adaptive immune system
    dY[4] = p*P - g*V - Bt*S*V               #virus
    return dY

def run1(tt,aS,dS,Bt,tau,dU,dP,k,aE,dE,E50,w,p,g,S0,E0):
    sol=odeint(model_1, [S0,U0,P0,E0,V0], tt, (aS,dS,Bt,tau,dU,dP,k,aE,dE,E50,w,p,g), mxstep=1000)
    return sol

#function to fit model 1 with pegged parameters
def fit1(tt,aS,dS,dU,dP,dE,E50,w,k,S0,E0):
    
    #initial conditions that are always true regardless of model
    U0=0      #no infected cells
    P0=0      #productively infected cells
    V0=0.03   #start with 30 copies per mL
    
    Bt=1e-4; tau=0.05; p=5e4; g=23; aE=1e-4; #pegged parameters
    
    sol=odeint(model_1, [S0,U0,P0,E0,V0], tt, (aS,dS,Bt,tau,dU,dP,k,aE,dE,E50,w,p,g), mxstep=1000)
    logV=np.log10(sol[:,4]*1e3) #log viral load copies per mL
    logV[logV<-3]=15 #safety net for bad parameter space

    return logV

# # same death rate for U & P
# (12 free parameters + 5 initial conditions)
# - now productive and unproductive have same natural death rates
def model_2(X,t,aS,dS,Bt,tau,dI,k,aE,dE,E50,w,p,g):  
    dY = np.zeros(5);
    S=X[0]; U=X[1]; P=X[2]; E=X[3]; V=X[4];    
    dY[0] = aS - dS*S - Bt*S*V               #susceptible cells
    dY[1] = (1-tau)*Bt*S*V - dI*U - k*E*U    #unproductively infected
    dY[2] = tau*Bt*S*V - dI*P - k*E*P        #productively infected
    dY[3] = w*E*(P+U)/(E+E50) + aE - dE*E;   #adaptive immune system
    dY[4] = p*P - g*V - Bt*S*V               #virus
    return dY

def run2(tt,aS,dS,Bt,tau,dI,k,aE,dE,E50,w,p,g,S0,E0):
    sol=odeint(model_2, [S0,U0,P0,E0,V0], tt, (aS,dS,Bt,tau,dI,k,aE,dE,E50,w,p,g), mxstep=1000)
    return sol

def fit2(tt,aS,dS,dE,E50,w,k,S0,E0):
    
    #initial conditions that are always true regardless of model
    U0=0      #no infected cells
    P0=0      #productively infected cells
    V0=0.03   #start with 30 copies per mL
    
    Bt=1e-4; dI=1; tau=0.05; p=5e4; g=23; aE=1e-4; #pegged parameters
    
    sol=odeint(model_2, [S0,U0,P0,E0,V0], tt, (aS,dS,Bt,tau,dI,k,aE,dE,E50,w,p,g), mxstep=1000)
    logV=np.log10(sol[:,4]*1e3) #log viral load copies per mL
    logV[logV<-3]=15 #safety net for bad parameter space

    return logV

# # adaptive does not respond to unproductive
# (12 free parameters + 5 initial conditions)
# - now the adaptive response does not remove U cells or grow proportionally to them, this is the same number of free parameters but tests this hypothesis. Technically wouldn't even have to simulate U at this point, but still need both beta and tau to make it work so that many susceptibles are removed even as not that many P are made.
def model_3(X,t,aS,dS,Bt,tau,dI,k,aE,dE,E50,w,p,g):  
    dY = np.zeros(5);
    S=X[0]; U=X[1]; P=X[2]; E=X[3]; V=X[4];    
    dY[0] = aS - dS*S - Bt*S*V               #susceptible cells
    dY[1] = (1-tau)*Bt*S*V - dI*U            #unproductively infected
    dY[2] = tau*Bt*S*V - dI*P - k*E*P        #productively infected
    dY[3] = w*E*P/(E+E50) + aE - dE*E;       #adaptive immune system
    dY[4] = p*P - g*V - Bt*S*V               #virus
    return dY

def run3(tt,aS,dS,Bt,tau,dI,k,aE,dE,E50,w,p,g,S0,E0):
    sol=odeint(model_3, [S0,U0,P0,E0,V0], tt, (aS,dS,Bt,tau,dI,k,aE,dE,E50,w,p,g), mxstep=1000)
    return sol

def fit3(tt,aS,dS,dE,E50,w,k,S0,E0):
    
    #initial conditions that are always true regardless of model
    U0=0      #no infected cells
    P0=0      #productively infected cells
    V0=0.03   #start with 30 copies per mL
    
    Bt=1e-4; dI=1; tau=0.05; p=5e4; g=23; aE=1e-4; #pegged parameters
    
    sol=odeint(model_3, [S0,U0,P0,E0,V0], tt, (aS,dS,Bt,tau,dI,k,aE,dE,E50,w,p,g), mxstep=1000)
    logV=np.log10(sol[:,4]*1e3) #log viral load copies per mL
    logV[logV<-3]=15 #safety net for bad parameter space

    return logV


# # no unproductive at all (beta = beta*tau)
# (11 free parameters + 4 initial conditions)
# - now the adaptive response does not remove U cells or grow proportionally to them, this is the same number of free parameters but tests this hypothesis. Technically wouldn't even have to simulate U at this point, but still need both beta and tau to make it work so that many susceptibles are removed even as not that many P are made.
def model_4(X,t,aS,dS,Bt,dI,k,aE,dE,E50,w,p,g):  
    dY = np.zeros(4);
    S=X[0]; P=X[1]; E=X[2]; V=X[3];    
    dY[0] = aS - dS*S - Bt*S*V               #susceptible cells
    dY[1] = Bt*S*V - dI*P - k*E*P        #productively infected
    dY[2] = w*E*P/(E+E50) + aE - dE*E;       #adaptive immune system
    dY[3] = p*P - g*V - Bt*S*V               #virus
    return dY

def run4(tt,aS,dS,Bt,dI,k,aE,dE,E50,w,p,g,S0,E0):
    sol=odeint(model_4, [S0,P0,E0,V0], tt, (aS,dS,Bt,dI,k,aE,dE,E50,w,p,g), mxstep=1000)
    return sol

def fit4(tt,aS,dS,dE,E50,w,k,S0,E0):
    
    #initial conditions that are always true regardless of model
    P0=0      #productively infected cells
    V0=0.03   #start with 30 copies per mL
    
    Bt=1e-4*0.05; dI=1; p=5e4; g=23; aE=1e-4; #pegged parameters
    
    sol=odeint(model_4, [S0,P0,E0,V0], tt, (aS,dS,Bt,dI,k,aE,dE,E50,w,p,g), mxstep=1000)
    logV=np.log10(sol[:,3]*1e3) #log viral load copies per mL
    logV[logV<-3]=15 #safety net for bad parameter space

    return logV


# # no adaptive
# (8 free parameters + 4 initial conditions)
# - now the adaptive response does not remove U cells or grow proportionally to them, this is the same number of free parameters but tests this hypothesis. Technically wouldn't even have to simulate U at this point, but still need both beta and tau to make it work so that many susceptibles are removed even as not that many P are made.
def model_5(X,t,aS,dS,Bt,tau,dU,dP,p,g):  
    dY = np.zeros(4);
    S=X[0]; U=X[1]; P=X[2]; V=X[3];    
    dY[0] = aS - dS*S - Bt*S*V               #susceptible cells
    dY[1] = (1-tau)*Bt*S*V - dU*U            #unproductively infected
    dY[2] = tau*Bt*S*V - dP*P        #productively infected
    dY[3] = p*P - g*V - Bt*S*V               #virus
    return dY

def run5(tt,aS,dS,Bt,tau,dU,dP,p,g,S0,E0):
    sol=odeint(model_5, [S0,U0,P0,V0], tt, (aS,dS,Bt,tau,dU,dP,p,g), mxstep=1000)
    return sol

def fit5(tt,aS,dS,S0):
    
    #initial conditions that are always true regardless of model
    U0=0      #no infected cells
    P0=0      #productively infected cells
    V0=0.03   #start with 30 copies per mL
    
    Bt=1e-4; dI=1; tau=0.05; p=5e4; g=23; #pegged parameters
    
    sol=odeint(model_5, [S0,U0,P0,V0], tt, (aS,dS,Bt,tau,dI,dI,p,g), mxstep=1000)
    logV=np.log10(sol[:,3]*1e3) #log viral load copies per mL
    logV[logV<-3]=15 #safety net for bad parameter space

    return logV

# # no adaptive and no undproductive
# (6 free parameters + 3 initial conditions)
# - now the adaptive response does not remove U cells or grow proportionally to them
def model_6(X,t,aS,dS,Bt,dI,p,g):  
    dY = np.zeros(3);
    S=X[0]; P=X[1]; V=X[2];    
    dY[0] = aS - dS*S - Bt*S*V  #susceptible cells
    dY[1] = Bt*S*V - dI*P   #productively infected
    dY[2] = p*P - g*V - Bt*S*V  #virus
    return dY

def run6(tt,aS,dS,Bt,dI,p,g,S0):
    sol=odeint(model_6, [S0,P0,V0], tt, (aS,dS,Bt,dI,p,g), mxstep=1000)
    return sol

def fit6(tt,aS,dS,S0):
    
    #initial conditions that are always true regardless of model
    P0=0      #productively infected cells
    V0=0.03   #start with 30 copies per mL
    
    Bt=1e-4*0.05; dI=1; p=5e4; g=23; #pegged parameters
    
    sol=odeint(model_6, [S0,P0,V0], tt, (aS,dS,Bt,dI,p,g), mxstep=1000)
    logV=np.log10(sol[:,2]*1e3) #log viral load copies per mL
    logV[logV<-3]=15 #safety net for bad parameter space

    return logV
