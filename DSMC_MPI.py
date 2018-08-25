
import numpy as np
import numpy.random as rd

m = 7 * 1.660538e-27 # kg
kB = 1.3806e-23 # J/K
T0 = 200.e-6 # K
alen = -1.45e-9 # m
U0Trap = 5.*kB*T0 # J
wx = 2.*np.pi*150.6 # rad/s
wy = 2.*np.pi*152.6 # rad/s
wz = 2.*np.pi*131.5 # rad/s
wAr = np.array([wx, wy, wz])

def U(r,w):
	Pot = 0.5 * m * ((w[0]**2)*(r[0]**2) + (w[1]**2)*(r[1]**2) + \
	(w[2]**2)*(r[2]**2))
	return Pot # J

def DU(r,w):	
	DPot = m * np.array([(w[0]**2)*r[0], (w[1]**2)*r[1], \
	(w[2]**2)*r[2]])
	return DPot # J/m
	
def ESC(Ri, Pi, dt):
	Q = Ri + (dt/(2.*m)) * Pi
	Pf = Pi - dt * DU(Q, wAr)
	Rf = Q + (dt/(2.*m)) * Pf
	return [Rf, Pf]
	
k = 51

Matrix = [[[[] for l in range(k)] for j in range(k)] for i in range(k)]

TMues = 1.0e-3 # m

MGT = (TMues)/((k-1)/2.) # m

"""
	Generación de posiciones y momentos.
"""

# Construimos los grupos de datos.

Np = 1000
Nm = 1000

PosMom = []

for i in range(0, Np):
	
	for i in range(0, Nm):
		
		rdx = rd.uniform(0.0, 1.0)
		rdy = rd.uniform(0.0, 1.0)
		rdz = rd.uniform(0.0, 1.0)
		
		pdx = rd.uniform(0.0, 1.0)
		pdy = rd.uniform(0.0, 1.0)
		pdz = rd.uniform(0.0, 1.0)
		
		ri = np.array([ss.erfinv(2.*rdx-1.)/wAr[0], \
		ss.erfinv(2.*rdy-1.)/wAr[1], ss.erfinv(2.*rdz-1.)/wAr[2]])
		
		pi = np.array([ss.erfinv(2.*pdx-1.), ss.erfinv(2.*pdy-1.), \
		ss.erfinv(2.*pdz-1.)])
		
		PosMom.append([((2.*kB*T0/m)**0.5)*ri,((2.*kB*T0*m)**0.5)*pi])
		
# Número de átomos de prueba.

N = Np*Nm

# Número de átomos físicos.

Nreal = 10000000

# Razón de átomos físicos y de prueba.

alp = Nreal/N

"""
	Colisiones.
"""

# Sección eficaz de dispersión.

def sig(vr):
	
	a0 = -1.45e-9 # Scattering length (m)
	sigma = 4. * np.pi * a0**2 # m^2
	
	return sigma

# Probabilidad de colisión entre dos partículas.

def MaxSV(AR):
	
	Ap = 0
	for i in range(0,len(AR)):
		for j in range(i,len(AR)):
			vr = abs(np2.norm(AR[i][1])-np2.norm(AR[j][1])) / m
			Ap1 = vr*sig(vr)
			if (Ap < Ap1):
				Ap = Ap1
	
	return [Ap, len(AR)]
	
# Densidad promedio en la celda.

def nc(AR):
	
	a = alp * len(AR) / (MGT**3)
	
	return a

def Pij(Ri, Rj, Max, Dt):

	Mc = Max[1]*(Max[1]-1)/2.
	vr = abs(np2.norm(Ri[1])-np2.norm(Rj[1])) / m
	res1 = alp * Dt * vr * sig(vr) / (MGT**3)
	res2 = mt.ceil(Mc* alp * Dt * Max[0] / (MGT**3)) / Mc

	return res1 / res2

"""
	Implementación de colisiones.
"""

t = 0
ArAp = np.array([TMues, TMues, TMues])

for i in range(0,len(PosMom)):
	
	if abs(PosMom[i][0][0])<=(TMues):
		if abs(PosMom[i][0][1])<=(TMues):
			if abs(PosMom[i][0][2])<=(TMues):
				
				Rn = 0
				Rn = (ArAp + PosMom[i][0]) / MGT
				Rn = np.array([mt.floor(Rn[0]), mt.floor(Rn[1]), \
				mt.floor(Rn[2])])
				Matrix[Rn[0]][Rn[1]][Rn[2]].append(PosMom[i])
	
# Time step.
Med = mt.floor(((k-1)/2.))
MaxM = MaxSV(Matrix[Med][Med][Med])
ncMax = nc(Matrix[Med][Med][Med])
for i in range(0,k):
	for j in range(0,k):
		for l in range(0,k):
			MaxM1 = MaxSV(Matrix[i][j][l])
			if (MaxM[0] < MaxM1[0]):
				MaxM = MaxM1
				ncMax = 1000*nc(Matrix[i][j][l])

tau = 1. / (ncMax * MaxM[0])

print(tau)

TStep = 0.1*tau

Matrix = [[[[] for l in range(k)] for j in range(k)] for i in range(k)]

PosMomS = []

for i in range(0,len(PosMom)): 
	if  (U0Trap >(U(PosMom[i][0],wAr)+ \
	(0.5*m*(np2.norm(PosMom[i][1])**2)))):
		PosMomS.append(PosMom[i])

Ener = []
Eap = 0.05
for i in range(0,161):
	
	Ener.append(Eap)
	Eap = Eap + 0.05

nEn = []

for i in range(0,161):
	
	nEn.append(0)

for i in range(0,len(PosMomS)):
	
	Ep = (U(PosMomS[i][0],wAr) + \
	(0.5*m*(np2.norm(PosMomS[i][1])**2))) / (kB*T0)
	nEnP = mt.floor(Ep/0.05)
	nEn[nEnP] = nEn[nEnP] + 1
	
with open('inicial2.dat', 'w') as f:
   writer = csv.writer(f, delimiter='\t')
   writer.writerows(zip(Ener,nEn)) 

print(len(PosMomS))

while (t<0.01):
	
	# Evolución sin colisión.
	for i in range(0,len(PosMomS)):
		PosMomS[i] = ESC(PosMomS[i][0], PosMomS[i][1], TStep)
		for j in range(0,3):
			if abs(PosMomS[i][0][j])>TMues:
				PosMomS[i][0][j] = (np.sign(PosMomS[i][0][j])*TMues - \
				PosMomS[i][0][j])
				PosMomS[i][1][j] = -PosMomS[i][1][j]
	
	# Almacenaje en la red.
	for i in range(0,len(PosMomS)):
	
		if abs(PosMomS[i][0][0])<=(TMues):
			if abs(PosMomS[i][0][1])<=(TMues):
				if abs(PosMomS[i][0][2])<=(TMues):
				
					Rn = 0
					Rn = (ArAp + PosMomS[i][0]) / MGT
					Rn = np.array([mt.floor(Rn[0]), mt.floor(Rn[1]), \
					mt.floor(Rn[2])])
					Matrix[Rn[0]][Rn[1]][Rn[2]].append(PosMomS[i])

	# Colisiones.
	for i in range(0, k):
		for j in range(0, k):
			for r in range(0, k):
				Max = MaxSV(Matrix[i][j][r])
				for l in range(0,len(Matrix[i][j][r])):
					for n in range(l+1,len(Matrix[i][j][r])):
						Rcol = rd.uniform(0.0, 1.0)
						if (Rcol<Pij(Matrix[i][j][r][l], 
						Matrix[i][j][r][n], Max, TStep)):
							
							Pm = ((Matrix[i][j][r][l][1] + \
							Matrix[i][j][r][n][1]) / 2.)
							Pr = Matrix[i][j][r][l][1] - \
							Matrix[i][j][r][n][1]
							pr = np2.norm(Pr)
							pd = np.sqrt(Pr[0]**2 + Pr[1]**2)
							
							the = np.arccos((1.-2.*rd.uniform(0.,1.)))
							phi = 2.*np.pi*rd.uniform(0.,1.)
							
							Prf = np.array([np.sin(the)* \
							((pr*Pr[1]*np.cos(phi)) + \
							(Pr[2]*Pr[0]*np.sin(phi))) + \
							pd*Pr[0]*np.cos(the), np.sin(the)*((Pr[1]*\
							Pr[2]*np.sin(phi)) - \
							(pr*Pr[0]*np.cos(phi))) + \
							pd*Pr[1]*np.cos(the), pd*(Pr[2]*np.cos(the) - \
							pd*np.sin(the)*np.sin(phi))])
							
							Prf = Prf / (2.*pd)
							
							Matrix[i][j][r][l][1] = Pm + Prf
							Matrix[i][j][r][n][1] = Pm - Prf
	
	PosMomS = []
	for i in range(0, k):
		for j in range(0, k):
			for r in range(0, k):
				for l in range(0,len(Matrix[i][j][r])):						
					PosMomS.append(Matrix[i][j][r][l])
	
	Matrix = [[[[] for l in range(k)] for j in range(k)] \
	for i in range(k)]
	
	t = t + TStep
	
	print(t,len(PosMomS))

nEn = []

for i in range(0,161):
	
	nEn.append(0)

for i in range(0,len(PosMomS)):
	
	Ep = (U(PosMomS[i][0],wAr) + \
	(0.5*m*(np2.norm(PosMomS[i][1])**2))) / (kB*T0)
	nEnP = mt.floor(Ep/0.05)
	nEn[nEnP] = nEn[nEnP] + 1
	
with open('final2.dat', 'w') as f:
   writer = csv.writer(f, delimiter='\t')
   writer.writerows(zip(Ener,nEn)) 

print('cortes')

t = 0.0

while (t<0.03):
	
	PosMomS2 = PosMomS
	
	PosMomS = []
	
	for i in range(0,len(PosMomS2)): 
		if  ((U0Trap*np.exp(-t*np.log(2)/(15*TStep)))>(U(PosMomS2[i][0],wAr)+ \
		(0.5*m*(np2.norm(PosMomS2[i][1])**2)))):
			PosMomS.append(PosMomS2[i])
	
	# Evolución sin colisión.
	for i in range(0,len(PosMomS)):
		PosMomS[i] = ESC(PosMomS[i][0], PosMomS[i][1], TStep)
		for j in range(0,3):
			if abs(PosMomS[i][0][j])>TMues:
				PosMomS[i][0][j] = (np.sign(PosMomS[i][0][j])*TMues - \
				PosMomS[i][0][j])
				PosMomS[i][1][j] = -PosMomS[i][1][j]
	
	# Almacenaje en la red.
	for i in range(0,len(PosMomS)):
	
		if abs(PosMomS[i][0][0])<=(TMues):
			if abs(PosMomS[i][0][1])<=(TMues):
				if abs(PosMomS[i][0][2])<=(TMues):
				
					Rn = 0
					Rn = (ArAp + PosMomS[i][0]) / MGT
					Rn = np.array([mt.floor(Rn[0]), mt.floor(Rn[1]), \
					mt.floor(Rn[2])])
					Matrix[Rn[0]][Rn[1]][Rn[2]].append(PosMomS[i])

	# Colisiones.
	for i in range(0, k):
		for j in range(0, k):
			for r in range(0, k):
				Max = MaxSV(Matrix[i][j][r])
				for l in range(0,len(Matrix[i][j][r])):
					for n in range(l+1,len(Matrix[i][j][r])):
						Rcol = rd.uniform(0.0, 1.0)
						if (Rcol<Pij(Matrix[i][j][r][l], 
						Matrix[i][j][r][n], Max, TStep)):
							
							Pm = ((Matrix[i][j][r][l][1] + \
							Matrix[i][j][r][n][1]) / 2.)
							Pr = Matrix[i][j][r][l][1] - \
							Matrix[i][j][r][n][1]
							pr = np2.norm(Pr)
							pd = np.sqrt(Pr[0]**2 + Pr[1]**2)
							
							the = np.arccos((1.-2.*rd.uniform(0.,1.)))
							phi = 2.*np.pi*rd.uniform(0.,1.)
							
							Prf = np.array([np.sin(the)* \
							((pr*Pr[1]*np.cos(phi)) + \
							(Pr[2]*Pr[0]*np.sin(phi))) + \
							pd*Pr[0]*np.cos(the), np.sin(the)*((Pr[1]*\
							Pr[2]*np.sin(phi)) - \
							(pr*Pr[0]*np.cos(phi))) + \
							pd*Pr[1]*np.cos(the), pd*(Pr[2]*np.cos(the) - \
							pd*np.sin(the)*np.sin(phi))])
							
							Prf = Prf / (2.*pd)
							
							Matrix[i][j][r][l][1] = Pm + Prf
							Matrix[i][j][r][n][1] = Pm - Prf
	
	PosMomS = []
	for i in range(0, k):
		for j in range(0, k):
			for r in range(0, k):
				for l in range(0,len(Matrix[i][j][r])):						
					PosMomS.append(Matrix[i][j][r][l])
	
	Matrix = [[[[] for l in range(k)] for j in range(k)] \
	for i in range(k)]
	
	t = t + TStep
	
	print(t,len(PosMomS))

nEn = []

for i in range(0,161):
	
	nEn.append(0)

for i in range(0,len(PosMomS)):
	
	Ep = (U(PosMomS[i][0],wAr) + \
	(0.5*m*(np2.norm(PosMomS[i][1])**2))) / (kB*T0)
	nEnP = mt.floor(Ep/0.05)
	nEn[nEnP] = nEn[nEnP] + 1
	
with open('final3.dat', 'w') as f:
   writer = csv.writer(f, delimiter='\t')
   writer.writerows(zip(Ener,nEn)) 

print('Sin cortes')

t = 0.0

while (t<0.01):
	
	# Evolución sin colisión.
	for i in range(0,len(PosMomS)):
		PosMomS[i] = ESC(PosMomS[i][0], PosMomS[i][1], TStep)
		for j in range(0,3):
			if abs(PosMomS[i][0][j])>TMues:
				PosMomS[i][0][j] = (np.sign(PosMomS[i][0][j])*TMues - \
				PosMomS[i][0][j])
				PosMomS[i][1][j] = -PosMomS[i][1][j]
	
	# Almacenaje en la red.
	for i in range(0,len(PosMomS)):
	
		if abs(PosMomS[i][0][0])<=(TMues):
			if abs(PosMomS[i][0][1])<=(TMues):
				if abs(PosMomS[i][0][2])<=(TMues):
				
					Rn = 0
					Rn = (ArAp + PosMomS[i][0]) / MGT
					Rn = np.array([mt.floor(Rn[0]), mt.floor(Rn[1]), \
					mt.floor(Rn[2])])
					Matrix[Rn[0]][Rn[1]][Rn[2]].append(PosMomS[i])

	# Colisiones.
	for i in range(0, k):
		for j in range(0, k):
			for r in range(0, k):
				Max = MaxSV(Matrix[i][j][r])
				for l in range(0,len(Matrix[i][j][r])):
					for n in range(l+1,len(Matrix[i][j][r])):
						Rcol = rd.uniform(0.0, 1.0)
						if (Rcol<Pij(Matrix[i][j][r][l], 
						Matrix[i][j][r][n], Max, TStep)):
							
							Pm = ((Matrix[i][j][r][l][1] + \
							Matrix[i][j][r][n][1]) / 2.)
							Pr = Matrix[i][j][r][l][1] - \
							Matrix[i][j][r][n][1]
							pr = np2.norm(Pr)
							pd = np.sqrt(Pr[0]**2 + Pr[1]**2)
							
							the = np.arccos((1.-2.*rd.uniform(0.,1.)))
							phi = 2.*np.pi*rd.uniform(0.,1.)
							
							Prf = np.array([np.sin(the)* \
							((pr*Pr[1]*np.cos(phi)) + \
							(Pr[2]*Pr[0]*np.sin(phi))) + \
							pd*Pr[0]*np.cos(the), np.sin(the)*((Pr[1]*\
							Pr[2]*np.sin(phi)) - \
							(pr*Pr[0]*np.cos(phi))) + \
							pd*Pr[1]*np.cos(the), pd*(Pr[2]*np.cos(the) - \
							pd*np.sin(the)*np.sin(phi))])
							
							Prf = Prf / (2.*pd)
							
							Matrix[i][j][r][l][1] = Pm + Prf
							Matrix[i][j][r][n][1] = Pm - Prf
	
	PosMomS = []
	for i in range(0, k):
		for j in range(0, k):
			for r in range(0, k):
				for l in range(0,len(Matrix[i][j][r])):						
					PosMomS.append(Matrix[i][j][r][l])
	
	Matrix = [[[[] for l in range(k)] for j in range(k)] \
	for i in range(k)]
	
	t = t + TStep
	
	print(t,len(PosMomS))

nEn = []

for i in range(0,161):
	
	nEn.append(0)

for i in range(0,len(PosMomS)):
	
	Ep = (U(PosMomS[i][0],wAr) + \
	(0.5*m*(np2.norm(PosMomS[i][1])**2))) / (kB*T0)
	nEnP = mt.floor(Ep/0.05)
	nEn[nEnP] = nEn[nEnP] + 1
	
with open('final4.dat', 'w') as f:
   writer = csv.writer(f, delimiter='\t')
   writer.writerows(zip(Ener,nEn)) 
