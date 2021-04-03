import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np

def func(x, a, b, c):
    return a * np.exp(-b * x) + c

xdata = np.array([13,20,27,34,41,48,55,62,69])
ydata = np.array([5.78294302048375, 5.536536960607981,5.477343847916859,5.3015912532488505,5.13099465488776,5.064981321840298,5.030092188335547,4.975724885775932,4.966869553371193])

plt.plot(xdata * 10**5, ydata, 'b*', label='data', color="dodgerblue")

popt, pcov = curve_fit(func, xdata, ydata)
popt

x_fit = np.linspace(0.8*min(xdata), 1.1*max(xdata))

plt.plot(x_fit * 10**5, func(x_fit, *popt), '-', color="mediumorchid",
         label=r'fit $ f_1: a=%5.2f, b=%5.2f, c=%5.2f$' % tuple(popt))

plt.xlabel(r'$N_{events}$')
plt.ylabel(r'$I_{68}$ (°)')
plt.title("68 % interval as a function of amount of events used for training")
plt.xlim([11.1 * 10**5, 73.6 * 10**5])
plt.legend()
plt.savefig('plots/F4_I68_overNevents.eps', format='eps')
# plt.show()
