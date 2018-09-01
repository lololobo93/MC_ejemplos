# Este programa simula el modelo de Ising bidimensional
# empleando el algoritmo de Wolf

import numpy as np
import numpy.random as rd
import pyprind as pp
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import rc, ticker
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

rc('font', **{'family': 'serif'})
rc('text', usetex=True)

eqSteps = 2000
n_Bins = 10
mcSteps = 100

nt = 19
T_list = np.linspace(4, 1.5, nt)
J = 1
N = 50
Nspins = N**2

Tc = 2.0/np.log(1.0 + np.sqrt(2))*J

animate = True
bw_cmap = colors.ListedColormap(['black', 'white'])

def wolf_met(config, padd, a, b, s):
    if (s == config[(a+1) % N, b]):
        if (rd.rand() < padd):
            config[(a+1) % N, b] *= -1
            wolf_met(config, padd, (a+1) % N, b, s)
    if (s == config[a, (b+1) % N]):
        if (rd.rand() < padd):
            config[a, (b+1) % N] *= -1
            wolf_met(config, padd, a, (b+1) % N, s)
    if (s == config[(a-1) % N, b]):
        if (rd.rand() < padd):
            config[(a-1) % N, b] *= -1
            wolf_met(config, padd, (a-1) % N, b, s)
    if (s == config[a, (b-1) % N]):
        if (rd.rand() < padd):
            config[a, (b-1) % N] *= -1
            wolf_met(config, padd, a, (b-1) % N, s)

def mcmove(config, padd):
    
    a = rd.randint(0, N)
    b = rd.randint(0, N)
    s =  config[a, b]
    config[a, b] *= -1
    wolf_met(config, padd, a, b, s)

def calcEnergy(config):
    energy = 0
    for i in range(len(config)):
        for j in range(len(config)):
            S = config[i,j]
            nb = -J*S*((config[(i+1)%N,j]) + 
            (config[i,(j+1)%N]) + 
            (config[(i-1)%N,j]) + 
            (config[i,(j-1)%N]))
            energy += nb
    return energy

def calcMag(config):
    mag = np.sum(config)
    return mag**2

Energy = np.zeros(nt)
Magnetization = np.zeros(nt)
SpecificHeat = np.zeros(nt)
Susceptibility = np.zeros(nt)

def initialstate(N):
    spins = np.zeros((N, N), dtype=np.int)
    for i in range(N):
        for j in range(N):
            spins[i, j] = 2*rd.randint(0,2) - 1
    return spins

# Corrida para cada temperatura
print("Corridas de temperatura:")
pbar = pp.ProgBar(len(T_list))
for m in range(len(T_list)):
    E1 = M1 = E2 = M2 = 0

# Equilibrar sistema
    config = initialstate(N)
    for _ in range(eqSteps):
        mcmove(config, 1-np.exp(-2*J/T_list[m]))   

    for _ in range(n_Bins):
        for l in range(mcSteps):
            mcmove(config, 1-np.exp(-2*J/T_list[m]))   
        
        Ene = calcEnergy(config)        
        Mag = calcMag(config)           

        E1 += Ene
        M1 += Mag
        M2 += Mag*Mag
        E2 += Ene*Ene

        if animate:
            plt.clf()
            plt.imshow(config, cmap=bw_cmap, norm=colors.BoundaryNorm([-1,0,1], bw_cmap.N))
            plt.xticks([])
            plt.yticks([])
            plt.title('%d x %d Modelo de Ising, T = %.3f' %(N,N,T_list[m]))
            plt.pause(0.01)

    Energy[m] = E1/(n_Bins*Nspins)
    Magnetization[m] = M1/(n_Bins*Nspins)
    SpecificHeat[m] = (E2/n_Bins - E1*E1/(n_Bins*n_Bins)) / \
        (Nspins*T_list[m]*T_list[m])
    Susceptibility[m] = (M2/n_Bins - M1*M1/(n_Bins*n_Bins))/(T_list[m])
    
    pbar.update()

plt.figure(figsize=(8, 6))

plt.subplot(221)
plt.axvline(x=Tc, color='k', linestyle='--')
plt.plot(T_list, Energy, 'o-')
plt.xlabel('$T$')
plt.ylabel('$<E>/N$')

plt.subplot(222)
plt.axvline(x=Tc, color='k', linestyle='--')
plt.plot(T_list, Magnetization, 'o-')
plt.xlabel('$T$')
plt.ylabel('$<M>/N$')

plt.subplot(223)
plt.axvline(x=Tc, color='k', linestyle='--')
plt.plot(T_list, SpecificHeat, 'o-')
plt.xlabel('$T$')
plt.ylabel('$C/N$')

plt.subplot(224)
plt.axvline(x=Tc, color='k', linestyle='--')
plt.plot(T_list, Susceptibility, 'o-')
plt.xlabel('$T$')
plt.ylabel('$\chi/N$')

plt.suptitle('%d x %d Modelo de Ising' % (N, N))

plt.show()
