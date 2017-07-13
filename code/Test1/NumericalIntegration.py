import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import time


def function(t, V0, RC):
    return V0*np.exp(-t/RC)


# determination of vars (V_0 , RC)
def const_calc(point_a, point_b):   # in format point_a = [t, v]
    RC = (point_b[0] - point_a[0])/np.log(point_a[1]/point_b[1])
    V0_a = point_a[1] * np.exp(point_a[0]/RC)
    V0_b = point_b[1] * np.exp(point_b[0] / RC)
    # print("RC = ", RC, "\nV0_a = ", V0_a, "    V0_b = ", V0_b, "\ndelta: ", V0_a-V0_b)
    return RC, V0_a


def calc_and_integrate(point_a, point_b, true_integral):
    RC, V0 = const_calc(point_a, point_b)
    return integrate.quad(function, 0, np.inf, args=(V0, RC))[0] - true_integral[0]


# plotting
def plot(V0, RC):
    time = np.linspace(0,10,2000)
    z = []
    for t in time:
        # fun
        z.append(function(t, V0, RC))

    plt.plot(time, z)
    plt.title('discharge')
    plt.xlabel('time')
    plt.ylabel('Volts')
    plt.show()


def error_numerical(true_RC, true_V0):
    true_integral = integrate.quad(function, 0, np.inf, args=(true_V0, true_RC))

    delta_v = 0    # assuming no error
    delta_t = 0.1

    t_a = 1
    t_b = 2
    V_a = function(t_a, true_V0, true_RC)
    V_b = function(t_b, true_V0, true_RC)

    t_list_a = np.linspace(t_a-delta_t, t_a+delta_t, 20)
    t_list_b = np.linspace(t_b-delta_t, t_b+delta_t, 20)

    T_a, T_b = np.meshgrid(t_list_a, t_list_b)
    zs = np.array([calc_and_integrate([a,V_a],[b, V_b],true_integral)
                   for a, b in zip(np.ravel(T_a), np.ravel(T_b))])
    integral = zs.reshape(T_a.shape)
    plt.figure()
    V = np.arange(-0.2,0.2,0.02)
    CS = plt.contour(T_a, T_b, integral, V)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.title('Contour Plot of deviation from true integration value at RC = ' + str('{0:.2f}'.format(true_RC)))
    plt.xlabel('t_a')
    plt.ylabel('t_b')
    plt.savefig("RC = " + str('{0:.2f}'.format(true_RC)) + ".png")
    plt.show()


def multiplot(startRC, endRC, stepRC):
    for RC in np.arange(startRC, endRC, stepRC):
        error_numerical(RC, 1)


def RCplot_single(RC = 1, V0 = 1):
    true_integral = integrate.quad(function, 0, np.inf, args=(V0, RC))
    st_a = 1
    delta_t = 0.1

    # trial points
    t_list_a = np.linspace(st_a - delta_t, st_a + delta_t, 20)
    v_a = function(st_a, V0, RC)
    t_b = 2
    v_b = function(t_b, V0, RC)

    # from points get RC & V0, integrate
    integrate_result = []

    for t_a in t_list_a:
        RC_, V0_ = const_calc([t_a,v_a],[t_b,v_b])
        integrate_result.append(integrate.quad(function, 0, np.inf, args=(V0_, RC_))[0] - true_integral[0] )

    # print(t_list_a, "\n", integrate_result)
    # plt.plot(t_list_a, integrate_result)
    # plt.show()

    return


def RCplota(true_V0):

    def calc(t_a, t_b, RC):
        V_a = function(1, true_V0, RC)
        V_b = function(2, true_V0, RC)
        true_integral = integrate.quad(function, 0, np.inf, args=(true_V0, RC))
        RC_, V0_ = const_calc([t_a,V_a], [t_b,V_b])
        # print(t_a)
        # print(RC,"new: ",RC_)
        result = integrate.quad(function, 0, np.inf, args=(V0_, RC_))[0] - true_integral[0]
        return result

    delta_v = 0  # assuming no error
    delta_t = 0.1

    t_a = 1
    t_b = 2

    t_list_a = np.linspace(t_a - delta_t, t_a + delta_t, 20)
    RC_list = np.linspace(0.1, 3, 100)

    T_a, RC = np.meshgrid(t_list_a, RC_list)
    zs = np.array([calc(a, t_b, RC)for a, RC in zip(np.ravel(T_a), np.ravel(RC))])

    integral = zs.reshape(T_a.shape)
    plt.figure()
    V = np.arange(-0.2, 0.2, 0.02)
    CS = plt.contour(T_a, RC, integral, V)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.title('Contour Plot of deviation from true integration value')
    plt.xlabel('t_a')
    plt.ylabel('RC')
    plt.savefig("RCplot.png")
    plt.show()


def RCplotb(true_V0):

    def calc(t_a, t_b, RC):
        V_a = function(1, true_V0, RC)
        V_b = function(2, true_V0, RC)
        true_integral = integrate.quad(function, 0, np.inf, args=(true_V0, RC))
        RC_, V0_ = const_calc([t_a,V_a], [t_b,V_b])
        # print(t_a)
        # print(RC,"new: ",RC_)
        result = integrate.quad(function, 0, np.inf, args=(V0_, RC_))[0] - true_integral[0]
        return result

    delta_v = 0  # assuming no error
    delta_t = 0.1

    t_a = 1
    t_b = 2

    t_list_b = np.linspace(t_b - delta_t, t_b + delta_t, 20)
    RC_list = np.linspace(0.1, 3, 100)

    T_b, RC = np.meshgrid(t_list_b, RC_list)
    zs = np.array([calc(t_a, b, RC)for b, RC in zip(np.ravel(T_b), np.ravel(RC))])

    integral = zs.reshape(T_b.shape)
    plt.figure()
    V = np.arange(-0.2, 0.2, 0.02)
    CS = plt.contour(T_b, RC, integral, V)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.title('Contour Plot of deviation from true integration value')
    plt.xlabel('t_b')
    plt.ylabel('RC')
    plt.savefig("RCplot.png")
    plt.show()


def prep_error(true_RC, true_V0):
    true_integral = integrate.quad(function, 0, np.inf, args=(true_V0, true_RC))

    t_a = 1
    t_b = 2
    v_a = function(t_a, true_V0, true_RC)
    v_b = function(t_b, true_V0, true_RC)
    error_list = np.arange(-10,10.01,0.5)  # errors in percentage

    integral_list = []
    for error in error_list:
        t_a_error = (t_a + (t_a * error / 100))
        t_b_error = (t_b + (t_b * error / 100))
        RC_, V0_= const_calc([t_a_error,v_a], [t_b_error,v_b])
        integral = integrate.quad(function, 0, np.inf, args=(V0_, RC_))
        integral_list.append(integral[0] - true_integral[0])

    plt.plot(error_list,integral_list)
    plt.show()


def prep_error2(true_V0):

    def calc(Error, RC):
        true_integral = integrate.quad(function, 0, np.inf, args=(true_V0, RC))

        t_a = true_t_a + true_t_a*(Error/100)
        t_b = true_t_b + true_t_b * (Error / 100)
        v_a = function(true_t_a, true_V0, RC)
        v_b = function(true_t_b, true_V0, RC)


        RC_, V0_ = const_calc([t_a, v_a], [t_b, v_b])
        result = integrate.quad(function, 0, np.inf, args=(V0_, RC_))[0] - true_integral[0]
        return result

    true_t_a = 1
    true_t_b = 2

    error_list = np.arange(-10,10.01,0.5)  # errors in percentage
    RC_list = np.linspace(-1, 3, 100)

    Error, RC = np.meshgrid(error_list, RC_list)
    zs = np.array([calc(E, RC) for E, RC in zip(np.ravel(Error), np.ravel(RC))])

    integral = zs.reshape(Error.shape)
    plt.figure()
    V = np.arange(-0.2, 0.201, 0.02)
    CS = plt.contour(Error, RC, integral, V)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.title('Contour Plot of deviation from true integration value')
    plt.xlabel('% error in timing')
    plt.ylabel('RC')
    plt.savefig("RCplot.png")
    plt.show()


def prep_error3(RC, true_V0):
    def calc(t_E, v_E):
        true_integral = integrate.quad(function, 0, np.inf, args=(true_V0, RC))

        t_a = true_t_a + true_t_a * (t_E / 100)
        t_b = true_t_b + true_t_b * (t_E / 100)
        v_a = true_v_a + true_v_a * (v_E / 100)
        v_b = true_v_b + true_v_b * (v_E / 100)

        RC_, V0_ = const_calc([t_a, v_a], [t_b, v_b])
        result = integrate.quad(function, 0, np.inf, args=(V0_, RC_))[0] - true_integral[0]
        return result

    true_t_a = 1
    true_t_b = 2
    true_v_a = function(true_t_a, true_V0, RC)
    true_v_b = function(true_t_b, true_V0, RC)
    # print("va:", true_v_a," vb: ", true_v_b)

    t_error_list = np.arange(-15, 15.01, 0.5)  # errors in percentage
    v_error_list = np.arange(-5, 5.01, 0.5)  # errors in percentage

    t_Error, v_Error = np.meshgrid(t_error_list, v_error_list)
    zs = np.array([calc(t_E, v_E) for t_E, v_E in zip(np.ravel(t_Error), np.ravel(v_Error))])

    integral = zs.reshape(t_Error.shape)
    plt.figure()
    V = np.arange(-0.2, 0.201, 0.02)
    CS = plt.contour(t_Error, v_Error, integral, V)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.title('Contour Plot of deviation from true integration value')
    plt.xlabel('% error in timing')
    plt.ylabel('% error in voltage')
    plt.title('Contour Plot of deviation from true integration value at RC = ' + str('{0:.2f}'.format(RC)))
    plt.savefig("RC = " + str('{0:.2f}'.format(RC)) + ".png")
    #plt.show()


def prep_error4(RC, true_V0):
    def calc(t_E, v_E):
        true_integral = integrate.quad(function, 0, np.inf, args=(true_V0, RC))

        t_a = true_t_a + true_t_a * (t_E / 100)
        t_b = true_t_b + true_t_b * (t_E / 100)
        v_a = true_v_a + v_E
        v_b = true_v_b + v_E

        RC_, V0_ = const_calc([t_a, v_a], [t_b, v_b])
        result = integrate.quad(function, 0, np.inf, args=(V0_, RC_))[0] - true_integral[0]
        return result

    true_t_a = 1
    true_t_b = 2
    true_v_a = function(true_t_a, true_V0, RC)
    true_v_b = function(true_t_b, true_V0, RC)
    # print("va:", true_v_a," vb: ", true_v_b)

    t_error_list = np.arange(-15, 15.01, 0.5)  # errors in percentage
    v_error_list = np.arange(-0.01, 0.0101, 0.001)  # absolute error /v

    t_Error, v_Error = np.meshgrid(t_error_list, v_error_list)
    zs = np.array([calc(t_E, v_E) for t_E, v_E in zip(np.ravel(t_Error), np.ravel(v_Error))])

    integral = zs.reshape(t_Error.shape)
    plt.figure()
    V = np.arange(-0.2, 0.201, 0.02)
    CS = plt.contour(t_Error, v_Error, integral, V)
    plt.clabel(CS, inline=1, fontsize=10)

    plt.xlabel('% error in timing')
    plt.ylabel('absolute error in voltage /v')
    plt.title('Contour Plot of deviation from true integration value at RC = ' + str('{0:.3f}'.format(RC)))
    plt.savefig("RC = " + str('{0:.3f}'.format(RC)) + ".png")
    plt.close()
    # plt.show()


def multiplot_prep(startRC, endRC, stepRC):
    for RC in np.arange(startRC, endRC, stepRC):
        print("\nCalculateing RC = ", RC,"...")
        start_time = time.time()
        prep_error4(RC, 1)
        print("--- %s seconds ---" % (time.time() - start_time))


multiplot_prep(0.4,0.6001,0.005)
#prep_error3(1,1)
#prep_error2(1)
# RCplota(1)
# RCplotb(1)

# multiplot(-0.1,0.21,0.02)
