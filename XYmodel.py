# Este programa simula el modelo XY bidimensional

import numpy as np
import numpy.random as rd
import pyprind as pp
import matplotlib.pyplot as plt
from matplotlib import rc, ticker
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

rc('font', **{'family': 'serif'})
rc('text', usetex=True)

eqSteps = 2000
mcSteps = 2000

def mcmove(config, beta):
    for i in range(N):
        for j in range(N):
                a = rd.randint(0, N)
                b = rd.randint(0, N)
                s = config[a, b]
                p = 2*np.pi*rd.uniform(0., 1.)
                nb1 = -(np.cos(config[(a+1) % N, b]-s) +
                        np.cos(config[a, (b+1) % N]-s) +
                        np.cos(config[(a-1) % N, b]-s) +
                        np.cos(config[a, (b-1) % N]-s))
                nb2 = -(np.cos(config[(a+1) % N, b]-p) +
                        np.cos(config[a, (b+1) % N]-p) +
                        np.cos(config[(a-1) % N, b]-p) +
                        np.cos(config[a, (b-1) % N]-p))
                cost = (nb2-nb1)
                if cost < 0:
                    config[a, b] = p
                elif rd.rand() < np.exp(-cost*beta):
                    config[a, b] = p
    return config


def calcEnergy(config):
    energy = 0
    for i in range(len(config)):
        for j in range(len(config)):
            S = config[i, j]
            nb = -(np.cos(config[(i+1) % N, j]-S) +
                   np.cos(config[i, (j+1) % N]-S) +
                   np.cos(config[(i-1) % N, j]-S) +
                   np.cos(config[i, (j-1) % N]-S))
            energy += nb
    return energy/4.

# Determina magnetizacion

def calcMag(config):
    mag = (np.sum(np.cos(config))**2 +
           np.sum(np.sin(config))**2)
    return mag


# Variables
nt = 20
N = 20

Energy = np.zeros(nt)
Magnetization = np.zeros(nt)
SpecificHeat = np.zeros(nt)
Susceptibility = np.zeros(nt)

T = np.linspace(0.01, 0.2, nt)  # Arreglo de temperatura

def initialstate(N):    # Generacion de estado
    state = 2*np.pi*rd.uniform(0.0, 1.0, size=(N, N))
    return state

# Corrida para cada temperatura
print("Corridas de temperatura:")
pbar = pp.ProgBar(len(T))
for m in range(len(T)):
    E1 = M1 = E2 = M2 = 0

# Equilibrar sistema
    config = initialstate(N)
    for i in range(eqSteps):
        mcmove(config, 1.0/T[m])

    for i in range(mcSteps):
        mcmove(config, 1.0/T[m])
        Ene = calcEnergy(config)
        Mag = calcMag(config)

        E1 = E1 + Ene
        M1 = M1 + Mag
        M2 = M2 + Mag*Mag
        E2 = E2 + Ene*Ene

        Energy[m] = E1/(mcSteps*N*N)
        Magnetization[m] = M1/(mcSteps*N*N)
        SpecificHeat[m] = (E2/mcSteps - E1*E1/(mcSteps*mcSteps))/(N*T[m]*T[m])
        Susceptibility[m] = (M2/mcSteps - M1*M1 /
                             (mcSteps*mcSteps))/(N*T[m]*T[m])

    pbar.update()

# Graficas
fig, ax = plt.subplots(2, 1)
ax[0].plot(T, Energy, 'kd--', alpha=0.8, lw=0.6)
# ax.set_xlim(0, 8)
# ax.set_ylim(0, 4)
# ax.grid(True, linestyle='--', alpha=0.3)
# ax[0].set_xlabel(r'$T$', fontsize=12)
ax[0].set_ylabel(r'$E$', fontsize=12)
ax[0].tick_params(which='major', direction='in', right=True, top=True)

ax[1].plot(T, abs(Magnetization), 'kd--', alpha=0.8, lw=0.6)
# ax.set_xlim(0, 8)
# ax.set_ylim(0, 4)
# ax.grid(True, linestyle='--', alpha=0.3)
ax[1].set_xlabel(r'$T$', fontsize=12)
ax[1].set_ylabel(r'$M$', fontsize=12)
ax[1].tick_params(which='major', direction='in', right=True, top=True)

plt.show()
