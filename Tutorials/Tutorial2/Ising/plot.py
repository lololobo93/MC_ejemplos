import numpy as np
import matplotlib.pyplot as plt

plt.subplot(211)

dataII=np.loadtxt('nnout.dat')
x=dataII[:,1]
y1=dataII[:,2]
y2=dataII[:,3]
c = plt.cm.Spectral((1)/4.,1)
plt.plot(x,y1,marker='D',markersize=5,color=c,linewidth=1,label='Low-T neuron')
c = plt.cm.Spectral((2)/4.,1)
plt.plot(x,y2,marker='s',markersize=5,color=c,linewidth=1,label='High-T neuron')

leg = plt.legend(loc='best',numpoints=1,markerscale=1.0,fontsize=15,labelspacing=0.1)

plt.ylabel('Average output layer', fontsize=15)
plt.xlabel(r'$T$', fontsize=15,labelpad=0)
plt.xlim([1,3.5383706284260401])

x=[2.26918,2.26918]
y=[0,1]
plt.plot(x, y,color='#FFA500',linewidth=2)

plt.subplot(212)

dataII=np.loadtxt('acc.dat')
x=dataII[:,1]
y1=dataII[:,2]
c = plt.cm.Spectral((3)/4.,1)
plt.plot(x,y1,marker='*',markersize=5,color=c,linewidth=1)

plt.ylabel('Average accuracy', fontsize=15)
plt.xlabel(r'$T$', fontsize=15,labelpad=0)
plt.xlim([1,3.5383706284260401])
plt.ylim([0.6,1])

x=[2.26918,2.26918]
y=[0.6,1]
plt.plot(x, y,color='#FFA500',linewidth=2)

plt.savefig('neuralnetoutput.pdf')

plt.show()

plt.subplot(111)

dataIII=np.loadtxt('hlout.dat')
x=dataIII[:,0]
y1=dataIII[:,1]
y2=dataIII[:,2]-4
y3=dataIII[:,3]
c = plt.cm.Spectral((3)/4.,1)
plt.scatter(x, y1)
plt.scatter(x, y2)
plt.scatter(x, y3)

plt.ylabel(r'$W x + b$', fontsize=15)
plt.xlabel(r'$T$', fontsize=15,labelpad=0)
# plt.xlim([-1, 1])

plt.savefig('hidenlayeroutput.pdf')

plt.show()