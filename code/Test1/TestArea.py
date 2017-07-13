import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
import matplotlib.mlab as mlab

x,y = np.mgrid[:1:1E-3,:1:1E-3]
xs = ((x-0.3)**2.)
ys = ((y-0.5)**2.)
z = np.exp(-1*(xs/0.5+ys/0.3))

pl.contourf(x,y,z,20)

a = np.array([1,2,3])
b = np.array([4,5,6])
c = np.array(a)
x = np.append(a, [a])

a = [1, 3, 5]
b = a
a[:] = [x + 2 for x in a]
print(b)
print(pow(2,3))

X = [[0,0],[1,0]]
Y = [[0,0],[0,1]]
Z = [[0,0],[1,1]]
plt.figure()
CS = plt.contour(X, Y, Z)
plt.clabel(CS, inline=1, fontsize=10)
plt.title('Contour Plot of Error Coeff. ^2')
plt.xlabel('V_b')
plt.ylabel('RC')
plt.show()


