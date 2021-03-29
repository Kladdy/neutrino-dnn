import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np

def func(x, a, b, c):
    return a * np.exp(-b * x) + c

xdata = np.array([69, 40, 20, 5])
ydata = np.array([5.107, 5.472, 6.005, 6.909])

plt.plot(xdata, ydata, 'b-', label='data')

popt, pcov = curve_fit(func, xdata, ydata)
popt

x_fit = np.linspace(min(xdata), max(xdata))

plt.plot(x_fit, func(x_fit, *popt), 'r-',
         label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))

plt.xlabel('number of training files')
plt.ylabel('68 % interval')
plt.legend()
plt.show()