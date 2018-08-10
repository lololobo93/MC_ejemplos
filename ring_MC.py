# Este programa simula un sistema de N esferas duras
# de diámetro D en un anillo de longitud L con condiciones
# periódicas.

import numpy as np
import numpy.random as rd
import pyprind as pp
import matplotlib.pyplot as plt
from matplotlib import rc, ticker
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

rc('font', **{'family': 'serif'})
rc('text', usetex=True)

# Variables
L = 16
N = 12
D = 1
dx = L/(D*N) - 1
EqStep = 2500
MCStep = 5000

# Posicion inicial
x = np.arange(0, L, L/(D*N))

def MC_sweep():
    xtrial = x[0] + dx*(rd.rand()-0.5)
    if ((xtrial - (x[N-1]-L))>D and (x[1] - xtrial)>D):
        x[0] = xtrial
    for i in range(1, N-1):
        xtrial = x[i] + dx*(rd.rand()-0.5)
        if ((xtrial - x[i-1])>D and (x[i+1] - xtrial)>D):
            x[i] = xtrial
    xtrial = x[N-1] + dx*(rd.rand()-0.5)
    if ((xtrial - x[N-2])>D and (x[0] + L - xtrial)>D):
        x[N-1] = xtrial

n_points = 100
dr = L/n_points
r = np.array([i*dr for i in range(0, n_points)])

def corr_calc():
    g = np.zeros(r.shape)
    for i in range(N):
        for j in range(i+1, N):
            Dx = abs(x[i] - x[j])
            if (Dx > 0.5*L):
                Dx = L - Dx
            ind = int(Dx/dr)
            g[ind] += 1
    return (g + g[::-1])/(N*dr)

# Sistema en equilibrio
for i in range(0, EqStep):
    MC_sweep()

print("Pasos Monte Carlo:")
pbar = pp.ProgBar(MCStep)
# Nuevos pasos
g_prom = np.zeros(r.shape)
for i in range(0, MCStep):
    MC_sweep()
    g_prom += corr_calc()/MCStep
    pbar.update()
    # print(g_prom[int(D/dr)])

fig, ax = plt.subplots()
plt.plot(r, g_prom, 'kd--', alpha = 0.8, lw = 0.6)
ax.set_xlim(0, 16)
ax.set_ylim(0, 2)
# ax.grid(True, linestyle='--', alpha=0.3)
ax.set_xlabel(r'$r$', fontsize=12)
ax.set_ylabel(r'$g(r)$', fontsize=12)
ax.tick_params(which='major', direction='in', right=True, top=True)
plt.show()
