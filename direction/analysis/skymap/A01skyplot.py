import numpy as np
from matplotlib import pyplot as plt
import peakutils
from scipy import signal
from radiotools import plthelpers as php
from radiotools import helper as hp
from radiotools import stats
import pickle
from astrotools.coord import rand_fisher_vec
from NuRadioReco.utilities import units
from astrotools import healpytools as hpt
from radiotools import coordinatesystems as cstrafo
import meander 
import astrotools.coord
from astrotools import auger, coord, skymap
from matplotlib import cm
import healpy
from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

nside = 32
n_pix = hpt.nside2npix(nside)


def compute_ang_err(ra, dec, sigma):
    r''' Compute circular uncertainty contour around point
    on a healpix map.
    Parameters:
    -----------
    ra: float
        Right ascension of center point
    dec: float
        declination of center point
    sigma:
        Angular uncertainty (radius of circle to draw
        around (ra,dec))
    Returns:
    --------
    Theta: array
        theta values of contour
    Phi: array
        phi values of contour
    '''

    dec = np.pi / 2 - dec
    sigma = np.rad2deg(sigma)
    delta, step, bins = 0, 0, 0
    delta = sigma / 180.0 * np.pi
    step = 1. / np.sin(delta) / 20.
    bins = int(360. / step)
    Theta = np.zeros(bins + 1, dtype=np.double)
    Phi = np.zeros(bins + 1, dtype=np.double)
    # define the contour
    for j in range(0, bins) :
            phi = j * step / 180.*np.pi
            vx = np.cos(phi) * np.sin(ra) * np.sin(delta) + np.cos(ra) * (np.cos(delta) * np.sin(dec) + np.cos(dec) * np.sin(delta) * np.sin(phi))
            vy = np.cos(delta) * np.sin(dec) * np.sin(ra) + np.sin(delta) * (-np.cos(ra) * np.cos(phi) + np.cos(dec) * np.sin(ra) * np.sin(phi))
            vz = np.cos(dec) * np.cos(delta) - np.sin(dec) * np.sin(delta) * np.sin(phi)
            idx = hp.vec2pix(nside, vx, vy, vz)
            DEC, RA = hp.pix2ang(nside, idx)
            Theta[j] = DEC
            Phi[j] = RA
    Theta[bins] = Theta[0]
    Phi[bins] = Phi[0]

    return Theta, Phi


def compute_contours(proportions, samples):
    r''' Plot containment contour around desired level.
    E.g 90% containment of a PDF on a healpix map

    Parameters:
    -----------
    proportions: list
        list of containment level to make contours for.
        E.g [0.68,0.9]
    samples: array
        array of values read in from healpix map
        E.g samples = hp.read_map(file)
    Returns:
    --------
    theta_list: list
        List of arrays containing theta values for desired contours
    phi_list: list
        List of arrays containing phi values for desired contours
    '''

    levels = []
    sorted_samples = list(reversed(list(sorted(samples))))
    nside = hpt.pixelfunc.get_nside(samples)
    sample_points = np.array(healpy.pix2ang(nside, np.arange(len(samples)))).T
    for proportion in proportions:
        level_index = (np.cumsum(sorted_samples) > proportion).tolist().index(True)
        level = (sorted_samples[level_index] + (sorted_samples[level_index + 1] if level_index + 1 < len(samples) else 0)) / 2.0
        levels.append(level)
    contours_by_level = meander.spherical_contours(sample_points, samples, levels)

    theta_list = []; phi_list = []
    for contours in contours_by_level:
        for contour in contours:
            theta, phi = contour.T
            phi[phi < 0] += 2.0 * np.pi
            theta_list.append(theta)
            phi_list.append(phi)

    return theta_list, phi_list


def get_nu_direction(l, p, theta):
    return np.sin(theta) * p / np.linalg.norm(p) + np.cos(theta) * l / np.linalg.norm(l)


def get_polarization(l, v):
    p = np.cross(l, np.cross(v, l))
    return p / np.linalg.norm(p)

def main():
    iD = 4
    for iZ, (zen_nu, az_nu) in enumerate(np.array([[140, 280],
                                [100, 100]]) * units.deg):
    # zen_nu = 140 * units.deg
    # az_nu = 230 * units.deg
        sl = 1 * units.deg
        sp = 8 * units.deg
        st = 4 * units.deg
        # sl = 0.00000001
        # sp = 0.00000001
        # st = 0.00000001
        
        N = 1000000
        # N = 100
        n_ice = 1.78
        theta_C = np.arccos(1. / n_ice)
        v = hp.spherical_to_cartesian(zen_nu, az_nu)
        l = hp.spherical_to_cartesian(zen_nu - theta_C, az_nu)
        p = get_polarization(l, v)
        cs = cstrafo.cstrafo(*hp.cartesian_to_spherical(*l))
        p_on_sky = cs.transform_from_ground_to_onsky(p)
        
        # Choose color map and set background to white
        cmap = cm.YlOrRd
        cmap.set_under("w")
        
        
        k = -1
        ls = ['--', ':', '-', '-', '-']
        colors = ['0.3', 'C2', 'C1', 'C0','k']
        for t in np.array([
                                    [1, 1000, 4],
                                    [1, 40, 4],
                                    [1, 8, 4],
                                    [1, 4, 4],
                                    [1, 4, 1],
                                    [0.2, 2, 1],
                                    ]) * units.deg:
        # for sl in np.array([1]) * units.deg:
        #     for sp in np.array([4, 8, 40]) * units.deg:
        #         for st in np.array([4]) * units.deg:
                    sl, sp, st = t
                    k+=1
                    ll = rand_fisher_vec(l, 1. / sl ** 2, N)
                    p_rot = np.random.normal(0, sp, N)
                    pp = np.zeros((3, N))
                    for iL in range(ll.shape[1]):
                        cs = cstrafo.cstrafo(*hp.cartesian_to_spherical(*ll[:,iL]))
                        pp_on_sky = np.roll(hp.rotate_vector_in_2d(np.roll(p_on_sky, -1), p_rot[iL]), 1)
                        pp[:, iL] = cs.transform_from_onsky_to_ground(pp_on_sky)
                        
        #             pp = rand_fisher_vec(p, 1. / sp ** 2, N)
                    tt = np.random.normal(theta_C, st, N)
                    
                    vv = -1 * get_nu_direction(ll, pp, tt)
                    thetas = astrotools.coord.angle(vv, -1 * v)
                    var = np.sum(thetas ** 2) / 2 / len(thetas)
                    sigma = var ** 0.5
                    q68 = sigma * (-2 * np.log(1 - 0.68)) ** 0.5
        #             fig, ax = php.get_histogram(thetas / units.deg, bins=np.arange(0, 40, 1), kwargs={'density':True, 'facecolor':'0.7', 'alpha':1, 'edgecolor':"k"})
        #             xx = np.linspace(0, 40, 10000)
        #             ax.plot(xx, xx / (var / units.deg ** 2) * np.exp(-xx ** 2 / (2 * var / units.deg ** 2)))
        #             plt.show()
                    
                    # calculate contour
                    # 1st bin in healpy
                    vvp = hpt.vec2pix(nside, *vv)
                    
                    h, bins = np.histogram(vvp, range(n_pix + 1), density=True)
                    # theta, phi = compute_contours([0.68], h)
                    # phi[phi>np.pi] -= 2 * np.pi
                    # az, zen = hpt.pix2ang(nside, range(n_pix))
                    
                    csum = np.cumsum(sorted(h, reverse=True))
                    A90 = np.argwhere(csum>0.90)[0][0] * hpt.nside2pixarea(nside)
                    A68 = np.argwhere(csum>0.68)[0][0] * hpt.nside2pixarea(nside)
                    print("uncertainties: signal direction = {:.1f}deg, polarization = {:.1f}deg, viewing angle = {:.1f}deg -> nu direction {:.1f}deg {:.1f}deg , A90 = {:.2f}, A68 = {:.2f}".format(sl / units.deg, sp / units.deg, st / units.deg, sigma / units.deg, q68 / units.deg, A90, A68))
                    
        #             hpt.mollview(h, cbar=True, rot=0, cmap=cmap, fig=0)
                    hpt.graticule()
                    
                    # zen, az = hp.cartesian_to_spherical(*vv)
                    zen_nu_tmp, az_nu_tmp = hp.cartesian_to_spherical(-v[0],-v[1], -v[2])
                    hpt.projscatter(zen_nu_tmp, az_nu_tmp, c='k', marker='x')
                    
                    # Draw containment contour around GW skymap
                    probs = hpt.pixelfunc.ud_grade(h, 16) #reduce nside to make it faster
                    probs = probs / np.sum(probs)
                    
                    levels = [0.90]
                    theta_contour, phi_contour = compute_contours(levels, probs)
                    if k < len(colors):
                        leg = r"$\sigma_l$ = {:.1f}$^\circ$, $\sigma_\theta$ = {:.1f}$^\circ$, $\sigma_p$ = {:.1f}$^\circ$".format(sl/units.deg, st/units.deg, sp/units.deg)
                        if(k == 0):
                            leg = r"$\sigma_l$ = {:.1f}$^\circ$, $\sigma_\theta$ = {:.1f}$^\circ$, $\sigma_p = \infty$".format(sl/units.deg, st/units.deg)
                        if(iZ == 0):
                            hpt.projplot(theta_contour[0], phi_contour[0], ls[k], color=colors[k], linewidth=3, label=leg)
                        else:
                            hpt.projplot(theta_contour[0], phi_contour[0], ls[k], color=colors[k], linewidth=3)
                        for i in range(1, len(theta_contour)):
                            hpt.projplot(theta_contour[i], phi_contour[i], ls[k], color=colors[k], linewidth=3)
                    
    # Add labels to plot
    plt.text(2.0, 0., r"$90^\circ$", ha="left", va="center")
    plt.text(1.9, 0.45, r"$60^\circ$", ha="left", va="center")
    plt.text(1.4, 0.8, r"$30^\circ$", ha="left", va="center")
    plt.text(1.9, -0.45, r"$120^\circ$", ha="left", va="center")
    plt.text(1.4, -0.8, r"$150^\circ$", ha="left", va="center")
    plt.text(1.333, -0.15, r"$60^\circ$", ha="center", va="center")
    plt.text(.666, -0.15, r"$120^\circ$", ha="center", va="center")
    plt.text(0.0, -0.15, r"$180^\circ$", ha="center", va="center")
    plt.text(-.666, -0.15, r"$240^\circ$", ha="center", va="center")
    plt.text(-1.333, -0.15, r"$300^\circ$", ha="center", va="center")
    plt.text(-2.0, -0.15, r"$360^\circ$", ha="center", va="center")

    plt.title('', fontsize=15)
    plt.legend(loc=1, bbox_to_anchor=(1.08, 1.09), prop={'size': 14})
    plt.savefig('plots/resolution_{:02d}.png'.format(iD),bbox_inches='tight')
    plt.show()

