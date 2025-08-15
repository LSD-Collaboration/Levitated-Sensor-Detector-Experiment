# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 14:39:39 2025

@author: shelb

Collection of functions for data analysis of psds from optically levitated
particles, including lorentzians fits, calibration, and temperature calculations

"""

import numpy as np
from scipy.optimize import curve_fit
from scipy import signal as s
from scipy.signal import welch
from scipy.constants import k as k_B
import matplotlib.pyplot as plt


def lorentzian(f, A, gamma, f0, offset):
    """simple lorentzian function"""
    return A * (gamma / 2) ** 2 / ((f - f0) ** 2 + (gamma / 2) ** 2) + offset


def lorentzian_psd(f, A, gamma, f0, offset):
    """single-sided psd for a harmonic oscillator to be used for calibrations"""
    return A * (gamma) / ((f0**2 - f**2) ** 2 + ((gamma) ** 2 * f**2)) + offset


# a thing to try: log based fitting
# would have to make an altered version of the guess_params function to go with this
# def lorentzian_psd_log(f, logA, loggamma, f0, offset):
#     A = 10**logA
#     gamma = 10**loggamma
#     return A * (gamma) / ((f0**2 - f**2)**2 + ((gamma)**2 * f**2)) + offset


def guess_params(freq, psd, print_results=True):
    """Calculates guesses for the fit of the lorentzian_psd() function for a
    subset of data centered around a peak"""
    # guess for center frequency
    i_pk = np.argmax(psd)
    f0_guess = freq[i_pk]

    # naive guess for gamma
    gamma_guess = 0.08 * (freq[-1] - freq[0])  # width of masked data with fudge factor

    # better guess for gamma ???
    # half_max = psd[i_pk] / 2
    # i_left = np.where(psd[:i_pk] < half_max)[0] # list of indices left of peak where psd < half_max
    # i_right = np.where(psd[i_pk:] < half_max)[0] # list of indices right of peak where psd < half_max
    # left = freq[i_left[-1]] if i_left.size else freq[0] # right-most frequency in i_left
    # right = freq[i_pk + i_right[0]] if i_right.size else freq[-1] # left-most frequency in i_right
    # print(f'left is {left} and right is {right}')
    # Q = f0_guess / (right - left) if (right - left) else 1e4
    # gamma_guess = f0_guess / Q # full width half max
    # this turns out to be way too narrow

    # naive guess for offset
    # offset_guess = np.min(psd)

    # better guess for offset
    offset_guess = np.median(psd[-50:])  # median of the noise floor at high frequencies

    # naive guesses for A
    # A_guess = np.max(psd) - np.min(psd) # height of whole data set

    # better guess for A
    A_guess = (
        np.max(psd) * gamma_guess * f0_guess**2
    )  # from setting freq=f0 in equation

    # alternative better guess for A ???
    # area = (psd[i_pk] - offset_guess) * (right - left) * np.pi / 2
    # A_guess = max(area / gamma_guess, 1e-30)
    # not good

    if print_results:
        print("  Guesses:")
        print(f"    Amplitude (A): {A_guess:.4e}")
        print(f"    Width (gamma): {gamma_guess:.2f} Hz")
        print(f"    Center freq (f0): {f0_guess:.2f} Hz")
        print(f"    Offset: {offset_guess:.4e}")

    return [A_guess, gamma_guess, f0_guess, offset_guess]


def fit_lorentzian(
    freq, psd, center_freq, fit_width=2000.0, print_results=True, plot=True
):
    """Takes in full data set and selects a subset centered around a given
    frequency, then fits the data with the lorentzian_psd() function using
    guesses from guess_params() function. Prints"""
    # select section of data to consider
    mask = (freq > center_freq - fit_width / 2) & (freq < center_freq + fit_width / 2)
    freq_data_masked = freq[mask]
    psd_data_masked = psd[mask]

    # change function choice here
    function = lorentzian_psd

    # fit
    p0 = guess_params(freq_data_masked, psd_data_masked)
    bound = ([0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf])

    # narrower bounds on parameters
    # lower_bound = [0.1*element for element in p0]
    # upper_bound = [10*element for element in p0]
    # bound = (lower_bound, upper_bound)
    # print(f"bounds = {bound}")

    popt, pcov, infodict, msg, ier = curve_fit(
        function,
        freq_data_masked,
        psd_data_masked,
        p0=p0,
        maxfev=1000000,
        xtol=3e-16,
        ftol=3e-16,
        gtol=3e-16,
        method="trf",
        full_output=True,
        bounds=bound,
    )
    perr = np.sqrt(np.diag(pcov))
    # plug fit results back into function for plotting
    psd_fit_curve = function(freq_data_masked, *popt)

    # calculate R^2 for the fit
    residuals = (psd_data_masked - psd_fit_curve) ** 2
    ss_res = sum(residuals)
    mean = np.mean(psd_data_masked)
    ss_tot = sum((psd_data_masked - mean) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    # number of times the function was called
    nfev = infodict["nfev"]

    if print_results:
        print("  Fit completed:")
        print(f"    Message: {msg}")
        print(f"    Number of function calls (nfev): {nfev}")
        print("  Fit Results:")
        print(f"    Amplitude (A): {popt[0]:.4e} ± {perr[0]:.4e}")
        print(f"    Width (gamma): {popt[1]:.2f} ± {perr[1]:.2f} Hz")
        print(f"    Center freq (f0): {popt[2]:.2f} ± {perr[2]:.2f} Hz")
        print(f"    Offset: {popt[3]:.4e} ± {perr[3]:.4e}")
        print(f"    R² of fit: {r2:.4f}")

    if plot:
        plot_fit(freq, psd, popt, perr, r2, freq_data_masked, psd_fit_curve)

    return popt, perr, r2, freq_data_masked, psd_fit_curve


def plot_fit(freq, psd, popt, perr, r2, freq_data_masked, psd_fit_curve):
    """plot generated in fit_lorentzian() function"""

    plt.figure(figsize=(8, 5))
    plt.semilogy(
        freq, psd, "k.", markersize=2, linestyle="solid", alpha=0.4, label="Data"
    )
    plt.semilogy(
        freq_data_masked,
        psd_fit_curve,
        "r-",
        linewidth=2,
        label=f"Fit: $f_0$={popt[2]:.2f} Hz\n$R^2$={r2:.4f}",
    )

    plt.xlabel("Frequency (Hz)", fontsize=14)
    plt.ylabel("PSD (V²/Hz)", fontsize=14)
    plt.title("Fit Results")  # , fontsize=16)

    plt.tick_params(axis="both", labelsize=12)
    plt.grid(True)
    plt.legend(fontsize=12)

    plt.xlim(-1e3, 3e4)
    plt.ylim(1e-15, 1e-3)

    plt.tight_layout()
    plt.show()

    return


# def filter_():

#     b_u, a_u = butter(4, [p_u['low'].get() * 1000, p_u['high'].get() * 1000], 'band', fs=fs_hz)
#     uncooled_filtered = filtfilt(b_u, a_u, uncooled_chunk[:][1])

#     return


def area_under_curve(freq, psd, center_freq, width, plot=True):
    """Calculate area under curve for subset of psd centered around a peak,
    make sure to used filter data"""
    mask = (freq > center_freq - width / 2) & (freq < center_freq + width / 2)
    freq_data_masked = freq[mask]
    psd_data_masked = psd[mask]

    area = np.trapz(psd_data_masked, freq_data_masked)

    # V_rms_sq_uncool = np.mean(uncooled_filtered**2)

    if plot:
        plot_area_under_curve(freq, psd, center_freq, width)

    return area


def plot_area_under_curve(freq, psd, center_freq, width):
    """plot generated in area_under_curve() function"""

    mask = (freq > center_freq - width / 2) & (freq < center_freq + width / 2)
    freq_data_masked = freq[mask]
    psd_data_masked = psd[mask]

    plt.figure(figsize=(8, 5))
    plt.semilogy(
        freq, psd
    )  # , 'k.', markersize=2, linestyle="solid", alpha=0.4, label="Data")
    # plt.plot(freq, psd)

    # plt.vlines(x=[center_freq-width/2,center_freq+width/2], ymin=1e-15, ymax=1e-3, color='lightblue', linewidth=2)
    plt.fill_between(freq_data_masked, psd_data_masked, 0, color="lightblue", alpha=0.4)

    plt.xlabel("Frequency (Hz)", fontsize=14)
    plt.ylabel("PSD (V²/Hz)", fontsize=14)
    plt.title("Area Under Curve")  # , fontsize=16)

    plt.tick_params(axis="both", labelsize=12)
    plt.grid(True)
    # plt.legend(fontsize=12)

    plt.xlim(center_freq - 2 * width, center_freq + 2 * width)
    # plt.xlim(-1e3,3e4)
    # plt.ylim(0,1e-7)

    plt.tight_layout()
    plt.show()

    # ahhh

    return


def particle_mass(shape, params):
    """Calculates mass of particle depending on shape
    For shape='sphere', params is (density,diameter)
    For shape='disc', params is (density,diameter,thickness)"""
    if shape == "sphere":
        # if len(params) != 2:
        #    raise ValueError('For a sphere, params = (density,diameter)')
        density = params[0]
        diameter = params[1]
        mass = density * (4 / 3) * np.pi * (diameter / 2) ** 3
    if shape == "disc":
        if len(params) != 3:
            raise ValueError("For a disc, params = (density,diameter,thickness)")
        density = params[0]
        diameter = params[1]
        thickness = params[2]
        mass = density * thickness * np.pi * (diameter / 2) ** 2
    else:
        raise ValueError("Acceptable shapes are 'sphere' or 'disc'")
    return mass


def find_calibration_coeff(
    freq, psd, center_freq, width, temp, particle_shape, shape_params
):
    """Takes in uncooled psd at high pressure to calculate calibration
    coefficient assumming equilibrium with the bath at room temp"""
    T_room = temp  # in K
    mass = particle_mass(particle_shape, shape_params)

    Vrms = area_under_curve(freq, psd, center_freq, width)

    calib = np.sqrt(mass * center_freq**2 * Vrms / (k_B * T_room))

    return calib


def temperature(freq, psd, center_freq, width, calib, particle_shape, shape_params):
    """Takes in cooled psd and calibration coefficient to calculate temperature"""
    mass = particle_mass(particle_shape, shape_params)

    Vrms = area_under_curve(freq, psd, center_freq, width)

    temp = mass * center_freq**2 * Vrms / (k_B * calib**2)

    return temp


def full_temperature_calibration(
    uncooled_data,
    cooled_data,
    sampling_rate,
    center_freq,
    width,
    particle_shape,
    shape_params,
    method="area_under_curve",
):
    """Does the full tempertaure calibration starting from cooled and uncooled
    time series data, and then generating the temperature of the cooled data"""

    fs = sampling_rate

    # compute the psds using periodogram
    uncooled_freq_p, uncooled_psd_p = s.periodogram(uncooled_data, fs)
    cooled_freq_p, cooled_psd_p = s.periodogram(cooled_data, fs)

    # plot
    plot_basic_cooling(uncooled_freq_p, uncooled_psd_p, cooled_freq_p, cooled_psd_p)

    if method == "area_under_curve":
        # do calibration with area under curve method
        # do I want this to still do a fit and use the fit parameters??

        room_temp = 300  # K
        particle_params = (4000, 6.2e-6, 190e-9)  # (density, diameter, thickness)

        # use welsh method to get smoothed psd
        NPERSEG = 1e4  # Number of samples per segment for Welch PSD
        uncooled_freq_w, uncooled_psd_w = welch(uncooled_data, fs=fs, nperseg=NPERSEG)
        cooled_freq_w, cooled_psd_w = welch(cooled_data, fs=fs, nperseg=NPERSEG)

        # use uncooled data to get the calibration coefficient
        calib = find_calibration_coeff(
            uncooled_freq_w,
            uncooled_psd_w,
            center_freq,
            width,
            room_temp,
            "disc",
            particle_params,
        )
        print(f"Calibration coefficient: {calib:.2e} V/m")

        # use cooled data to get the cooled temperature, using the calib from above
        temp = temperature(
            cooled_freq_w,
            cooled_psd_w,
            center_freq,
            width,
            calib,
            "disc",
            particle_params,
        )
        print(f"Cooled temperature: {temp:.1f} K")

    if method == "from_fit_params":
        return
        # !!! need to add this

    return


def plot_basic_cooling(uncooled_freq, uncooled_psd, cooled_freq, cooled_psd):
    """plot generated in full_temperature_calibration() function"""
    plt.figure(figsize=(8, 5))

    plt.semilogy(
        uncooled_freq, uncooled_psd, color="crimson", alpha=0.85, label="uncooled"
    )
    plt.semilogy(
        cooled_freq, cooled_psd, color="lightskyblue", alpha=0.85, label="cooled"
    )

    plt.legend()
    plt.xlabel("Frequency (Hz)", fontsize=14)
    plt.ylabel("PSD (V²/Hz)", fontsize=14)

    plt.xlim(-1e3, 3e4)
    plt.ylim(1e-15, 1e-3)

    # plt.title(f"folder: {FOLDER}")
    plt.grid(True, which="both", ls="--", lw=0.5)


# TODO

# alt methods - temp from area under fit, temp from fit parameters

# decide if i want to make them all take raw data or psds

# make plots better and more consistent in axis limits
