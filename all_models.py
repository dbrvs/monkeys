
# coding: utf-8

# # Nested models
# 
# In this script, I have all the models that I use in the nested model analysis. These go from most complicated and most free parameters, down to least complicated and least free parameters. Each heading describes the feature that was removed to make the model less complicated.
# 
# The cheat sheet is as follows:
# 
# 1. complete
# 2. same natural death rate U and P
# 3. E doesn't see U cells
# 4. E does not saturate (back to E seeing U, and remains true henceforth)
# 5. simplified E formulation
# 6. no E at all
# 

import numpy as np

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
    dY[3] = w*(P+U)*E/(1+E/E50) + aE - dE*E;   #adaptive immune system
    dY[4] = p*P - g*V - Bt*S*V               #virus
    return dY

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

# # adaptive does not saturate
# (11 free parameters + 5 initial conditions)
# - now there is no saturation in adaptive compartment
def model_4(X,t,aS,dS,Bt,tau,dI,k,aE,dE,w,p,g):  
    dY = np.zeros(5);
    S=X[0]; U=X[1]; P=X[2]; E=X[3]; V=X[4];    
    dY[0] = aS - dS*S - Bt*S*V               #susceptible cells
    dY[1] = (1-tau)*Bt*S*V - dI*U - k*E*U    #unproductively infected
    dY[2] = tau*Bt*S*V - dI*P - k*E*P        #productively infected
    dY[3] = w*E*(P+U) + aE - dE*E; #adaptive immune system
    dY[4] = p*P - g*V - Bt*S*V               #virus
    return dY

# # simplified adaptive
# (9 free parameters + 3 initial conditions)
# - now the adaptive response is non-mechanistic, just kills infecteds based on how many productively infected cells are around
# - here have just ignored U completed except for the impact of tau
def model_5(X,t,aS,dS,Bt,tau,dI,k,P50,p,g):  
    dY = np.zeros(3);
    S=X[0]; P=X[1]; V=X[2];    
    dY[0] = aS - dS*S - Bt*S*V               #susceptible cells
    dY[1] = tau*Bt*S*V - dI*P - k*P/(P+P50)  #productively infected
    dY[2] = p*P - g*V - Bt*S*V               #virus
    return dY

# # no adaptive at all
# (7 free parameters + 3 initial conditions)
# - now the adaptive response does not remove U cells or grow proportionally to them
def model_6(X,t,aS,dS,Bt,tau,dI,p,g):  
    dY = np.zeros(3);
    S=X[0]; P=X[1]; V=X[2];    
    dY[0] = aS - dS*S - Bt*S*V  #susceptible cells
    dY[1] = tau*Bt*S*V - dI*P   #productively infected
    dY[2] = p*P - g*V - Bt*S*V  #virus
    return dY
