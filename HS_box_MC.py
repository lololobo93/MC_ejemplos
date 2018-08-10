# Este programa simula un sistema de N esferas duras
# de diámetro D en una caja bidimensional de LxL.

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
n = 12
N = n**2
D = 1
dx = L/(D*n) - 1
EqStep = 500
MCStep = 1000

# Posicion inicial
x = np.arange(0, L, L/(D*n))
y = np.arange(0, L, L/(D*n))
xx, yy = np.meshgrid(x,  y)
# plt.scatter(xx.ravel(),yy.ravel())

def MC_sweep():
    for _ in range(N):
        x_i = rd.randint(0, n)
        y_i = rd.randint(0, n)
        xtrial = xx[x_i, y_i] + dx*(rd.rand()-0.5)
        ytrial = yy[x_i, y_i] + dx*(rd.rand()-0.5)
        if (xtrial > L):
            xtrial -= L
        if (ytrial > L):
            ytrial -= L
        if (xtrial < 0):
            xtrial += L
        if (ytrial < 0):
            ytrial += L
        collision = False
        for i in range(n):
            for j in range(n):
                if (i != x_i or j != y_i):
                    Dx = abs(xx[i, j] - xtrial)
                    Dy = abs(yy[i, j] - ytrial)
                    if (Dx > 0.5*L):
                        Dx = L - Dx
                    if (Dy > 0.5*L):
                        Dy = L - Dy
                    if (np.sqrt(Dx*Dx + Dy*Dy) < D):
                        collision = True
                if collision:
                    break
            if collision:
                break
        if not collision:
            xx[x_i, y_i] = xtrial
            yy[x_i, y_i] = ytrial

n_points = 100
dr = L/(2*n_points)
r = np.array([i*dr for i in range(n_points)])

def corr_calc():
    g = np.zeros(r.shape)
    XY_list = list(zip(xx.ravel(), yy.ravel()))
    for i in range(0, N):
        for j in range(i+1, N):
            Dx = abs(XY_list[i][0] - XY_list[j][0])
            Dy = abs(XY_list[i][1] - XY_list[j][1])
            if (Dx > 0.5*L):
                Dx = L - Dx
            if (Dy > 0.5*L):
                Dy = L - Dy
            ind = int(np.sqrt(Dx*Dx + Dy*Dy)/dr)
            if ind < n_points:
                g[ind] += 1/r[ind]
    return g/(N*dr)


# Sistema en equilibrio
for i in range(0, EqStep):
    MC_sweep()

# plt.scatter(xx.ravel(), yy.ravel())

# Nuevos pasos
print("Pasos Monte Carlo:")
pbar = pp.ProgBar(MCStep)
g_prom = np.zeros(r.shape)
for i in range(0, MCStep):
    MC_sweep()
    g_prom += corr_calc()/MCStep
    pbar.update()

# plt.scatter(xx.ravel(), yy.ravel())

fig, ax = plt.subplots()
ax.plot(r, g_prom, 'kd--', alpha=0.8, lw=0.6)
ax.set_xlim(0, 8)
ax.set_ylim(0, 4)
# ax.grid(True, linestyle='--', alpha=0.3)
ax.set_xlabel(r'$r$', fontsize=12)
ax.set_ylabel(r'$g(r)$', fontsize=12)
ax.tick_params(which='major', direction='in', right=True, top=True)
plt.show()
