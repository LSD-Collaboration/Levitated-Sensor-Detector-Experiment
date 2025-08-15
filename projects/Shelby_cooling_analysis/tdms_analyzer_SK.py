# -*- coding: utf-8 -*-
"""
Created on Jul 2025

@author: shelb

Reads in data in tdsm format and does lots of plotting and analysis based on
functions in analysis_functions.py

"""

# %%

import os
from nptdms import TdmsFile
from scipy import signal as s
from scipy.signal import welch
import matplotlib.pyplot as plt

import analysis_functions as fun  # homemade functions for fitting, calibrating, etc.


def tdsm_import(files, CHANNEL_INDEX, FILE_INDEX, FS=None):
    """Takes list of all file names, and single indices for channel and file
    to import single dataset and its sampling frequency"""

    # Select the m-th file (1-based index)
    m_idx = FILE_INDEX - 1
    if not (0 <= m_idx < len(files)):
        raise IndexError(
            f"FILE_INDEX must be between 1 and {len(files)}, got {FILE_INDEX}"
        )
    tdms_path = files[m_idx]
    print(f"> Opening file #{FILE_INDEX}: {tdms_path}")

    # Read the TDMS file and select the only group
    td = TdmsFile(tdms_path)
    groups = td.groups()
    if not groups:
        raise RuntimeError("No groups found in TDMS file.")
    if len(groups) > 1:
        print(f"  ⚠️  Multiple groups found; using the first: {groups[0].name}")
    grp = groups[0]

    # Select the n-th channel (1-based index)
    channels = grp.channels()
    if not channels:
        raise RuntimeError(f"No channels found in group '{grp.name}'.")
    n_idx = CHANNEL_INDEX
    if not (0 <= n_idx < len(channels)):
        raise IndexError(
            f"CHANNEL_INDEX must be between 0 and {len(channels)}, got {CHANNEL_INDEX}"
        )
    ch = channels[n_idx]
    print(f"> Processing channel #{CHANNEL_INDEX}: {ch.name}")
    data = ch[:]  # Load channel data into a NumPy array

    # Determine sampling frequency
    # FS can set sampling frequency in Hz; if None, reads 'wf_increment' from channel properties
    if FS is not None:
        fs = FS
        print(f"Using user-specified FS = {fs} Hz")
    else:
        props = ch.properties
        if "wf_increment" in props:
            fs = 1.0 / props["wf_increment"]
            print(f"Read wf_increment={props['wf_increment']}; FS = {fs} Hz")
        else:
            raise ValueError(
                "Sampling rate not provided and 'wf_increment' not in channel properties."
            )

    return (data, fs)


# %% short version with fits and calibration

plt.close("all")

# Gather and sort list of all .tdms files in the folder
path = "C:/Users/shelb/Desktop/"
FOLDER = path + "Shelby_071725_hexcooling_02"

files = sorted(
    os.path.join(FOLDER, f) for f in os.listdir(FOLDER) if f.lower().endswith(".tdms")
)
if not files:
    raise FileNotFoundError(f"No .tdms files found in {FOLDER!r}")


# list of 1-based indexes of the TDMS files to process
FILE_INDEXES = [8, 52]  # 8,52 or 187,245
FILE_NAMES = ["no cooling", "cooling"]

# list of 0-based indexes of the channels to analyze
CHANNEL_INDEXES = [2]
CHANNEL_NAMES = ["Bottom QPDx"]  # 2

uncooled_data, fs_unc = tdsm_import(files, CHANNEL_INDEXES[0], FILE_INDEXES[0])
cooled_data, fs_c = tdsm_import(files, CHANNEL_INDEXES[0], FILE_INDEXES[1])

sampling_rate = fs_c
center_freq = 17360  # 400, 21860, 17360 # Hz
width = 3000  # Hz
particle_shape = "disc"
shape_params = (4000, 6.2e-6, 190e-9)  # (density, diameter, thickness)


# just fitting
uncooled_freq, uncooled_psd = s.periodogram(uncooled_data, fs_unc)
cooled_freq, cooled_psd = s.periodogram(cooled_data, fs_c)

print("**Fitting uncooled case**")
(
    uncooled_popt,
    uncooled_perr,
    uncooled_r2,
    uncooled_freq_data_masked,
    uncooled_psd_fit_curve,
) = fun.fit_lorentzian(uncooled_freq, uncooled_psd, center_freq, width)
print("**Fitting cooled case**")
cooled_popt, cooled_perr, cooled_r2, cooled_freq_data_masked, cooled_psd_fit_curve = (
    fun.fit_lorentzian(cooled_freq, cooled_psd, center_freq, width)
)


# this does it all
cooled_temp = fun.full_temperature_calibration(
    uncooled_data,
    cooled_data,
    sampling_rate,
    center_freq,
    width,
    particle_shape,
    shape_params,
)


# %% earlier version: read in data, calculate psds, save to dict, and plot


# Gather and sort list of all .tdms files in the folder
path = "C:/Users/shelb/Desktop/"
FOLDER = path + "\Shelby_071725_hexcooling_02"

files = sorted(
    os.path.join(FOLDER, f) for f in os.listdir(FOLDER) if f.lower().endswith(".tdms")
)
if not files:
    raise FileNotFoundError(f"No .tdms files found in {FOLDER!r}")


# list of 1-based indexes of the TDMS files to process
FILE_INDEXES = [8, 52]  # 8,52 or 187,245
FILE_NAMES = ["no cooling", "cooling"]

# list of 0-based indexes of the channels to analyze
CHANNEL_INDEXES = [2]
CHANNEL_NAMES = ["Bottom QPDx"]
# "Bottom QPD y" "Fiber output Detector" "Box signal into AOM"

plt.close("all")

psd_data = {index: {} for index in CHANNEL_NAMES}

for CHANNEL_INDEX, CHANNEL_NAME in zip(CHANNEL_INDEXES, CHANNEL_NAMES):

    plt.figure(figsize=(8, 5))

    psd_data[CHANNEL_NAME] = {index: {} for index in FILE_NAMES}

    for FILE_INDEX, FILE_NAME in zip(FILE_INDEXES, FILE_NAMES):

        data, fs = tdsm_import(files, CHANNEL_INDEX, FILE_INDEX)

        # Compute the PSD using periodogram
        freqs_p, Pxx_p = s.periodogram(data, fs)

        psd_data[CHANNEL_NAME][FILE_NAME]["frequencies"] = freqs_p
        psd_data[CHANNEL_NAME][FILE_NAME]["psd"] = Pxx_p

        # alternatively use welsh method to get smoothed psd
        NPERSEG = 1e4  # Number of samples per segment for Welch PSD
        freqs_w, Pxx_w = welch(data, fs=fs, nperseg=NPERSEG)

        psd_data[CHANNEL_NAME][FILE_NAME]["welsh_frequencies"] = freqs_w
        psd_data[CHANNEL_NAME][FILE_NAME]["welsh_psd"] = Pxx_w

        # 7) Plot results

        if FILE_NAME == "cooling":
            c = "lightskyblue"
        elif FILE_NAME == "no cooling":
            c = "crimson"
        else:
            c = "lightgray"

        # plt.figure(figsize=(8, 5))
        plt.semilogy(
            freqs_p,
            Pxx_p,
            color=c,
            alpha=0.9,
            label=f"file #{FILE_INDEX}, channel: {CHANNEL_NAME}",
        )  # , using scipy.signal.periodogram")
        # plt.semilogy(freqs_w, Pxx_w, label=f"file #{FILE_INDEX}, channel {CHANNEL_NAME}") #,  using scipy.signal.welsh")

    plt.legend()
    plt.xlabel("Frequency (Hz)", fontsize=14)
    plt.ylabel("PSD (V²/Hz)", fontsize=14)

    plt.xlim(-1e3, 3e4)
    plt.ylim(1e-15, 1e-3)

    plt.title(f"folder: {FOLDER}")
    plt.grid(True, which="both", ls="--", lw=0.5)

    plt.tight_layout()


OUTPUT = (
    None  # Path to save the PSD plot (e.g. "psd.png"); if None, displays interactively
)
if OUTPUT:
    plt.tight_layout()
    plt.savefig(OUTPUT, dpi=300)
    print(f"✅ Saved PSD plot to {OUTPUT}")
else:
    plt.show()


# %% earlier version: fits and calibrations


plot_output_path = None  # Or a path like "output.png"

plt.close("all")

for CHANNEL_INDEX, CHANNEL_NAME in zip(CHANNEL_INDEXES, CHANNEL_NAMES):

    center_freq = 400  # 21860 # 17360  # Hz
    width = 800  # Hz

    uncooled_freq = psd_data[CHANNEL_NAME]["no cooling"]["frequencies"]
    uncooled_psd = psd_data[CHANNEL_NAME]["no cooling"]["psd"]

    uncooled_freq_w = psd_data[CHANNEL_NAME]["no cooling"]["welsh_frequencies"]
    uncooled_psd_w = psd_data[CHANNEL_NAME]["no cooling"]["welsh_psd"]

    cooled_freq = psd_data[CHANNEL_NAME]["cooling"]["frequencies"]
    cooled_psd = psd_data[CHANNEL_NAME]["cooling"]["psd"]

    cooled_freq_w = psd_data[CHANNEL_NAME]["cooling"]["welsh_frequencies"]
    cooled_psd_w = psd_data[CHANNEL_NAME]["cooling"]["welsh_psd"]

    # Fit near center frequency
    # popt, perr, r2, freq_data_masked, psd_fit_curve = fun.fit_lorentzian_near_peak(freq , psd, center_freq, fit_width)

    # Fit near center frequency
    print("**fitting uncooled case**")
    (
        uncooled_popt,
        uncooled_perr,
        uncooled_r2,
        uncooled_freq_data_masked,
        uncooled_psd_fit_curve,
    ) = fun.fit_lorentzian(uncooled_freq, uncooled_psd, center_freq, width)
    print("**fitting cooled case**")
    (
        cooled_popt,
        cooled_perr,
        cooled_r2,
        cooled_freq_data_masked,
        cooled_psd_fit_curve,
    ) = fun.fit_lorentzian(cooled_freq, cooled_psd, center_freq, width)

    # grab center frequencies and gammas from fit
    fit_center_freq_unc = uncooled_popt[2]
    fit_center_freq_c = cooled_popt[2]

    fit_width_unc = uncooled_popt[1]
    fit_width_c = cooled_popt[1]

    # do calibration with area under curve method
    room_temp = 300  # K
    particle_params = (4000, 6.2e-6, 190e-9)  # (density, diameter, thickness)

    # use uncooled data to get the calibration coefficient
    calib = fun.find_calibration_coeff(
        uncooled_freq_w,
        uncooled_psd_w,
        fit_center_freq_unc,
        20 * fit_width_unc,
        room_temp,
        "disc",
        particle_params,
    )
    print(f"Calibration coefficient: {calib:.2e} V/m")

    # use cooled data to get the cooled temperature, using the calib from above
    temp = fun.temperature(
        cooled_freq_w,
        cooled_psd_w,
        fit_center_freq_c,
        20 * fit_width_c,
        calib,
        "disc",
        particle_params,
    )
    print(f"Cooled temperature: {temp:.1f} K")

    # makes plots to see if using reasonable width and smoothing

    # and make comparison with fitting method


# if plot_output_path:
#     plt.savefig(plot_output_path, dpi=300)
#     print(f"Plot saved to: {plot_output_path}")
# else:
#     plt.show()
