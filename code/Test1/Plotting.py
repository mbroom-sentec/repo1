import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def fun(a, b):
    T_sq = 1
    max_val = 2000
    z = (T_sq + (np.log(a / b)) ** -2) * ((b ** 2 + a ** 2) / (a * b) ** 2)
    if z > max_val:
        z = float('NaN')
    return z

def contour2d():
    delta = 0.025
    x = np.arange(0, 2.0, delta)
    y = np.arange(0, 2.0, delta)
    X, Y = np.meshgrid(x, y)
    zs = np.array([fun(x, y) for x, y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)

    V = []

    def contour_levels(n, spacing):
        i = 0
        level = 0
        while i < n:
            V.append(2 ** level)

            level += spacing
            i += 1

    contour_levels(20, 0.5)

    plt.figure()
    CS = plt.contour(X, Y, Z, V)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.title('Contour Plot of Error Coeff. ^2')
    plt.xlabel('V_a')
    plt.ylabel('V_b')
    plt.show()

def plot2d(a = 1):
    b = np.linspace(0,2,200)
    z = []
    for b_val in b:
        z.append(fun(a,b_val))
    plt.plot(b,z)
    plt.title('plot of ''b'' against Error Coeff. (V_a = 1)')
    plt.xlabel('V_b')
    plt.ylabel('Error Coeff')
    plt.show()

def fun_RCa(v_a, RC):
    delta_T = 1
    t_a = 1
    max_val = 2000
    z = ((t_a/delta_T)**2 + (RC/delta_T)**2)*((np.exp(2*delta_T/RC)+1)/v_a**2) + v_a**-2
    if z > max_val:
        z = float('NaN')
    return z

def plot2d_RC(v_a = 1):
    RC = np.linspace(0,10,200)
    z = []
    for RC_val in RC:
        # fun
        z.append(fun_RCa(v_a,RC_val))
    plt.plot(RC,z)
    plt.title('plot of ''RC'' against Error Coeff. (V_a = 1)')
    plt.xlabel('RC')
    plt.ylabel('Error Coeff')
    plt.show()


def contour2d_RC():
    delta = 0.025
    x = np.arange(0, 2.0, delta)
    y = np.arange(0, 8.0, delta)
    X, Y = np.meshgrid(x, y)
    # fun
    zs = np.array([fun_RCa(x, y) for x, y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)

    V = []

    def contour_levels(n, spacing):
        i = 0
        level = 0
        while i < n:
            V.append(2 ** level)

            level += spacing
            i += 1

    contour_levels(20, 0.5)

    plt.figure()
    CS = plt.contour(X, Y, Z, V)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.title('Contour Plot of Error Coeff. ^2')
    plt.xlabel('V_b')
    plt.ylabel('RC')
    plt.show()



plot2d_RC()
contour2d_RC()
plot2d()
contour2d()


