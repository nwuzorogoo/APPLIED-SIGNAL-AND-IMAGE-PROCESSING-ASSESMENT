"""Figures for the report. Each function draws one figure from a finished Study."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from mpl_toolkits.axes_grid1 import make_axes_locatable  # noqa: E402
from scipy import signal  # noqa: E402

from . import config  # noqa: E402
from . import convolution as conv  # noqa: E402
from . import evaluation as ev  # noqa: E402
from . import filters as flt  # noqa: E402
from .names import STRESS_NAMES  # noqa: E402
from .spectral import amplitude_spectrum  # noqa: E402
from .timefreq import dwt_single_level_reconstructions, stft_magnitude  # noqa: E402

BLUE, ORANGE, GREEN, GREY, INK, PURPLE = (
    "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#0b0b0b", "#4a3aa7")
SERIES = (GREEN, BLUE, ORANGE, PURPLE)
BAND_SYMBOLS = {"delta": "δ", "theta": "θ", "alpha": "α", "beta": "β", "gamma": "γ"}
STYLE = {
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 7.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e6e3",
    "grid.linewidth": 0.5, "lines.linewidth": 0.9, "font.family": "DejaVu Sans",
}
LEGEND_SMALL = dict(frameon=False, loc="lower left", bbox_to_anchor=(0.08, 0.0), fontsize=6,
                    title_fontsize=6, handlelength=1.0, labelspacing=0.25, borderaxespad=0.2)


def _save(fig, out_dir, name, dpi=300):
    fig.tight_layout()
    fig.savefig(Path(out_dir) / name, dpi=dpi)
    plt.close(fig)


def _zp_db(sos, fs, n_points=2 ** 15):
    w, h = flt.frequency_response(sos, fs, n_points)
    return w, flt.to_zero_phase_db(h)


def time_and_spectra(study, out_dir):
    """Figure 1: raw records with detected events shaded, and custom-FFT spectra."""
    fig, ax = plt.subplots(3, 2, figsize=(7.2, 3.7), gridspec_kw={"width_ratios": [1.35, 1]})
    for row, i in enumerate(study.ids):
        a = ax[row, 0]
        a.plot(study.time, study.x[i], color=BLUE, lw=0.5)
        spans = [(p.onset_s, p.offset_s) for p in study.pulses[i]] + [study.bursts[i][:2]]
        for start, end in spans:
            a.axvspan(start, end, color=ORANGE, alpha=0.15, lw=0)
        a.set_ylabel(f"DS{i} (µV)")
        a.set_xlim(0, study.duration)
        a.set_ylim(-110, 150)
        f, amp = study.spectrum(study.x[i], config.PLOT_NFFT)
        b = ax[row, 1]
        for k, (lo, hi) in enumerate(config.BANDS.values()):
            b.axvspan(lo, hi, color=GREY, alpha=0.08 if k % 2 else 0.16, lw=0)
        b.plot(f, amp, color=BLUE, lw=0.7)
        b.set_xlim(0, config.CLINICAL_CEILING_HZ)
        b.set_ylim(0, 27)
        for fp, ap in study.main_peaks[i]:
            b.annotate(f"{fp:.1f}", (fp, ap), xytext=(2, 1), textcoords="offset points",
                       fontsize=6.5, color=INK)
        b.set_ylabel("Amplitude (µV)")
    ax[0, 1].set_title("Amplitude spectrum (custom radix-2 FFT)")
    ax[0, 0].set_title("Raw EEG, shaded = detected pulses and theta burst")
    for name, (lo, hi) in config.BANDS.items():
        ax[0, 1].text((lo + hi) / 2, 25.0, BAND_SYMBOLS[name], ha="center", fontsize=6.5,
                      color=GREY)
    ax[2, 0].set_xlabel("Time (s)")
    ax[2, 1].set_xlabel("Frequency (Hz)")
    _save(fig, out_dir, "fig1_time_freq.png")


def aliasing(study, results, out_dir):
    """Figure 2: naive against anti-aliased decimation of the alias record."""
    factor = config.ALIAS_DEMO_FACTOR
    fs_new, naive, filtered = study.decimated(factor)
    x = study.x[config.ALIAS_RECORD]
    demo = results["alias_demo"]
    beta, folded = study.alias_probe["beta"], study.alias_probe["folded"]
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.0), gridspec_kw={"width_ratios": [1.2, 1, 1]})
    seg = (study.time >= 1) & (study.time < 1.5)
    t_new = np.arange(naive.size) / fs_new
    seg_new = (t_new >= 1) & (t_new < 1.5)
    ax[0].plot(study.time[seg], x[seg], color=BLUE, lw=0.8, label=f"{study.fs:g} Hz original")
    ax[0].plot(t_new[seg_new], naive[seg_new], "o-", color=ORANGE, ms=2.5, lw=0.8,
               label=f"{fs_new:g} Hz naive")
    ax[0].plot(t_new[seg_new], filtered[seg_new], "s-", color=GREEN, ms=2.5, lw=0.8,
               label=f"{fs_new:g} Hz anti-aliased")
    ax[0].set_xlabel("Time (s)")
    ax[0].set_ylabel("µV")
    ax[0].legend(loc="lower center", ncol=3, frameon=False, fontsize=5.6, handlelength=1.2,
                 columnspacing=0.8)
    ax[0].set_ylim(-80, 50)
    ax[0].set_title(f"(a) DS{config.ALIAS_RECORD} segment")
    f0, a0 = study.spectrum(x, config.PLOT_NFFT)
    ax[1].plot(f0, a0, color=BLUE, lw=0.7)
    ax[1].axvline(fs_new / 2, color=GREY, ls="--", lw=0.8)
    ax[1].text(fs_new / 2 + 0.5, 17, f"new Nyquist\n{fs_new / 2:g} Hz", fontsize=6.5, color=GREY)
    ax[1].set_xlim(0, 35)
    ax[1].set_ylim(0, 23)
    ax[1].set_title("(b) Original spectrum")
    ax[1].set_xlabel("Frequency (Hz)")
    ax[1].set_ylabel("µV")
    f_n, a_n = amplitude_spectrum(naive, fs_new, config.ALIAS_NFFT)
    f_a, a_a = amplitude_spectrum(filtered, fs_new, config.ALIAS_NFFT)
    ax[2].plot(f_n, a_n, color=ORANGE, lw=0.8, label="naive")
    ax[2].plot(f_a, a_a, color=GREEN, lw=0.8, label="anti-aliased")
    ax[2].annotate(f"{beta:.1f} Hz\nfolded to {folded:.1f}", (folded, demo["naive_amp_at_2p6"]),
                   xytext=(folded + 0.3, 19.3), fontsize=6.5, ha="center", va="bottom", color=INK)
    ax[2].set_xlim(0, fs_new / 2)
    ax[2].set_ylim(0, 25)
    ax[2].legend(frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.42), fontsize=6.5)
    ax[2].set_title(f"(c) After ↓{factor}, fs = {fs_new:g} Hz")
    ax[2].set_xlabel("Frequency (Hz)")
    _save(fig, out_dir, "fig2_aliasing.png")


def convolution(study, out_dir):
    """Figure 3: kernels, their responses and their effect on the reference pulse."""
    kernels = study.kernels()
    x3 = study.x[config.TRANSIENT_RECORD]
    beta = study.probe(config.ALIAS_RECORD, "beta")
    labels = {name: (f"Gaussian σ={sigma * 1e3:.1f} ms" if sigma else f"Moving avg. {len(h)} taps")
              for name, (h, sigma) in kernels.items()}
    colours = dict(zip(kernels, (BLUE, GREEN, ORANGE)))
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.0), gridspec_kw={"width_ratios": [0.8, 1, 1.25]})
    for name, (h, _) in kernels.items():
        lag_ms = (np.arange(len(h)) - (len(h) - 1) / 2) / study.fs * 1e3
        ax[0].plot(lag_ms, h, "o-", ms=1.8, color=colours[name], lw=0.8, label=labels[name])
        w, resp = signal.freqz(h, worN=4096, fs=study.fs)
        ax[1].plot(w, 20 * np.log10(np.abs(resp) + 1e-9), color=colours[name], lw=0.9)
    ax[0].set_xlabel("Lag (ms)")
    ax[0].set_title("(a) Kernels h[k]")
    ax[0].legend(frameon=False, fontsize=6, loc="upper right")
    ax[0].set_xlim(-40, 40)
    ax[1].axvline(beta, color=GREY, ls=":", lw=0.8)
    ax[1].text(beta + 0.6, -55, f"{beta:.1f} Hz β", fontsize=6.5, color=GREY)
    ax[1].set_ylim(-60, 3)
    ax[1].set_xlim(0, study.nyquist)
    ax[1].set_xlabel("Frequency (Hz)")
    ax[1].set_ylabel("|H| (dB)")
    ax[1].set_title("(b) Kernel frequency response")
    onset = study.pulse.onset_s
    zoom = study.mask((onset - 0.14, onset + 0.26))
    t = study.time[zoom]
    ax[2].plot(t, x3[zoom], color=GREY, lw=1.1, label="raw")
    styles = {"gauss40": ("Gauss 40 Hz", "-"), "gauss15": ("Gauss 15 Hz", "-"),
              f"ma{config.MOVING_AVERAGE_TAPS}": (f"MA-{config.MOVING_AVERAGE_TAPS}", "--")}
    for name, (h, _) in kernels.items():
        label, ls = styles[name]
        ax[2].plot(t, conv.convolve_same(x3, h)[zoom], color=colours[name], lw=0.9, ls=ls,
                   label=label)
    ax[2].set_xlabel("Time (s)")
    ax[2].set_ylabel("µV")
    ax[2].legend(frameon=False, ncol=2, fontsize=6, loc="lower right")
    ax[2].set_title(f"(c) DS{config.TRANSIENT_RECORD} pulse at {onset:.1f} s after convolution")
    _save(fig, out_dir, "fig3_convolution.png")


def time_frequency(study, out_dir):
    """Figure 4: two STFT windows against the Morlet CWT, detected pulses dotted."""
    x3 = study.x[config.TRANSIENT_RECORD]
    w, fr = study.cwt_map
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.25), sharey=True)
    meshes = []
    for a, nper, tag in zip(ax[:2], config.STFT_WINDOWS, "ab"):
        f, ts, z = stft_magnitude(x3, study.fs, nper)
        meshes.append(a.pcolormesh(ts, f, 20 * np.log10(z + 1e-3), shading="gouraud",
                                   cmap="Blues", vmin=-15, vmax=25, rasterized=True))
        a.set_title(f"({tag}) STFT, {nper / study.fs:g} s (Δf = {study.fs / nper:g} Hz)")
        a.set_xlabel("Time (s)")
    cwt_mesh = ax[2].pcolormesh(study.time, fr, 20 * np.log10(w + 1e-3), shading="gouraud",
                                cmap="Blues", vmin=0, vmax=40, rasterized=True)
    # cone of influence: values within one edge e-folding time of either end are unreliable
    coi = study.cwt.cone_of_influence_s(fr)
    duration = study.duration
    ax[2].fill_betweenx(fr, 0, np.minimum(coi, duration / 2), color="white", alpha=0.55, lw=0)
    ax[2].fill_betweenx(fr, np.maximum(duration - coi, duration / 2), duration, color="white",
                        alpha=0.55, lw=0)
    ax[2].plot(np.minimum(coi, duration / 2), fr, color=INK, lw=0.7, ls="--")
    ax[2].plot(np.maximum(duration - coi, duration / 2), fr, color=INK, lw=0.7, ls="--")
    ax[2].set_xlim(0, duration)
    ax[2].set_title("(c) CWT, complex Morlet")
    ax[2].set_xlabel("Time (s)")
    ax[0].set_ylabel("Frequency (Hz)")
    ax[0].set_ylim(0, config.CLINICAL_CEILING_HZ)
    for a in ax:
        a.grid(False)
        for pulse in study.pulses[config.TRANSIENT_RECORD]:
            a.axvline(pulse.centre_s, color=ORANGE, lw=0.5, ls=":")
    for a, mesh, label in ((ax[1], meshes[1], "STFT (dB)"), (ax[2], cwt_mesh, "CWT (dB)")):
        cax = make_axes_locatable(a).append_axes("right", size="4%", pad=0.06)
        bar = fig.colorbar(mesh, cax=cax)
        bar.set_label(label, fontsize=7.5)
        bar.ax.tick_params(labelsize=7)
    _save(fig, out_dir, "fig4_tf.png")


def dwt(study, results, out_dir):
    """Appendix figure: single-level DWT reconstructions of the transient record."""
    x3 = study.x[config.TRANSIENT_RECORD]
    parts = dwt_single_level_reconstructions(x3)
    fig, ax = plt.subplots(len(parts) + 1, 1, figsize=(7.2, 5.2), sharex=True)
    ax[0].plot(study.time, x3, color=INK, lw=0.5)
    ax[0].set_ylabel(f"DS{config.TRANSIENT_RECORD}", fontsize=6.5)
    for a, (name, (lo, hi)), part in zip(ax[1:], results["dwt_bands"].items(), parts):
        a.plot(study.time, part, color=BLUE, lw=0.5)
        a.set_ylabel(f"{name} {lo:g}–{hi:g} Hz", fontsize=6)
    ax[-1].set_xlabel("Time (s)")
    _save(fig, out_dir, "figA_dwt.png", dpi=250)


def filter_parameters(study, out_dir):
    """Figure 5: effect of order and cut-off, and the passband detail of the two designs."""
    fs = study.fs
    fig, ax = plt.subplots(1, 4, figsize=(7.2, 2.1))
    sweeps = (
        ("(a) Butterworth, order N", "N", config.BUTTER_ORDERS,
         lambda o: flt.butter_bandpass(o, fs)),
        (f"(b) Butterworth N={config.DESIGN_ORDER}, fc", "fc (Hz)", config.BUTTER_CUTOFFS,
         lambda c: flt.butter_bandpass(config.DESIGN_ORDER, fs, high=c)),
        ("(c) Elliptic, order N", "N", config.ELLIP_ORDERS, lambda o: flt.ellip_bandpass(o, fs)),
    )
    for a, (title, legend_title, values, design) in zip(ax, sweeps):
        for colour, value in zip(SERIES, values):
            w, db = _zp_db(design(value), fs)
            a.plot(w, db, color=colour, label=f"{value}")
        a.set_title(title)
        a.legend(title=legend_title, **LEGEND_SMALL)
        a.set_xlim(0, 100)
        a.set_ylim(-120, 5)
        a.set_xlabel("Frequency (Hz)")
        a.axvline(config.MAINS_HZ, color=GREY, ls=":", lw=0.7)
    ax[0].set_ylabel("Zero-phase |H|² (dB)")
    for sos, colour, label in ((study.sos_butter, BLUE, "Butterworth"),
                               (study.sos_ellip, ORANGE, "Elliptic")):
        w, db = _zp_db(sos, fs)
        ax[3].plot(w, db, color=colour, label=label)
    ax[3].set_xlim(0, config.CLINICAL_CEILING_HZ)
    ax[3].set_ylim(-6.5, 0.5)
    ax[3].set_title("(d) Passband detail")
    ax[3].set_xlabel("Frequency (Hz)")
    ax[3].legend(frameon=False, loc="lower center", fontsize=6, handlelength=1.0)
    _save(fig, out_dir, "fig5_filter_params.png")


def filter_performance(study, out_dir):
    """Figure 6: one stress-test draw, its PSD, pulse ringing and single-pass group delay."""
    fs, rec = study.fs, config.TRANSIENT_RECORD
    x3 = study.x[rec]
    xc = ev.contaminate(x3, fs, np.random.default_rng(ev.draw_seed(rec, 0)))
    y_b, y_e = study.zp(study.sos_butter, xc), study.zp(study.sos_ellip, xc)
    y_bn = study.zp(study.sos_butter_notch, xc)
    fig = plt.figure(figsize=(7.2, 3.3))
    grid = fig.add_gridspec(2, 3, height_ratios=[1, 1])
    a = fig.add_subplot(grid[0, :2])
    a.plot(study.time, xc, color=GREY, lw=0.4, label=f"DS{rec} + drift + 50 Hz + EMG")
    a.plot(study.time, y_b, color=BLUE, lw=0.6, label=f"Butterworth N={config.DESIGN_ORDER}")
    a.plot(study.time, y_e, color=ORANGE, lw=0.6, label=f"Elliptic N={config.DESIGN_ORDER}",
           alpha=0.9)
    a.set_xlim(0, study.duration)
    a.set_ylabel("µV")
    a.legend(frameon=False, ncol=3, loc="upper left", fontsize=6.5)
    a.set_title(f"(a) Stress test on contaminated DS{rec} (one draw, zero-phase)")
    a.set_ylim(-120, 190)
    b = fig.add_subplot(grid[0, 2])
    for z, colour, label, ls in ((xc, GREY, "contaminated", "-"), (y_b, BLUE, "Butterworth", "-"),
                                 (y_e, ORANGE, "Elliptic", "-"),
                                 (y_bn, GREEN, "Butterworth + notch", "--")):
        f, psd = signal.welch(z, fs, nperseg=512)
        b.semilogy(f, psd, color=colour, lw=0.7, ls=ls, label=label)
    b.set_xlim(0, study.nyquist)
    b.set_ylim(1e-6, 1e6)
    b.set_xlabel("Frequency (Hz)")
    b.set_ylabel("PSD (µV²/Hz)")
    b.set_title("(b) Welch PSD")
    b.legend(frameon=False, fontsize=5.3, loc="upper right", ncol=2, columnspacing=0.6,
             handlelength=1.2)
    c = fig.add_subplot(grid[1, :2])
    onset = study.pulse.onset_s
    zoom = study.mask((onset - 0.19, onset + 0.36))
    c.plot(study.time[zoom], x3[zoom], color=GREY, lw=1.2, label=f"raw DS{rec}")
    c.plot(study.time[zoom], study.zp(study.sos_butter, x3)[zoom], color=BLUE, lw=0.9,
           label="Butterworth")
    c.plot(study.time[zoom], study.zp(study.sos_ellip, x3)[zoom], color=ORANGE, lw=0.9,
           label="Elliptic")
    c.set_xlabel("Time (s)")
    c.set_ylabel("µV")
    c.set_title(f"(c) Pulse artefact at {onset:.1f} s: ringing and pre-ringing (zero-phase)")
    c.legend(frameon=False, fontsize=6.5, loc="upper right")
    d = fig.add_subplot(grid[1, 2])
    band = np.linspace(0.6, config.CLINICAL_CEILING_HZ, 600)
    for sos, colour, label in ((study.sos_butter, BLUE, "Butterworth"),
                               (study.sos_ellip, ORANGE, "Elliptic")):
        d.plot(band, flt.group_delay(sos, band, fs) / fs * 1e3, color=colour, label=label)
    d.set_yscale("log")
    d.set_xlabel("Frequency (Hz)")
    d.set_ylabel("Group delay (ms)")
    d.set_title("(d) Group delay, single pass")
    d.legend(frameon=False, fontsize=6)
    _save(fig, out_dir, "fig6_filter_perf.png")


def notch_and_stress(study, out_dir):
    """Figure 7: notch responses and the spread of stress-test SNR gain."""
    fs = study.fs
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.2), gridspec_kw={"width_ratios": [1, 1.4]})
    for q, colour in zip(config.NOTCH_Q_STUDY, (GREEN, BLUE, ORANGE)):
        w, db = _zp_db(flt.notch(fs, q=q), fs, 2 ** 18)
        ax[0].plot(w, db, color=colour, label=f"Q = {q:g}")
    w, db = _zp_db(study.sos_butter_notch, fs, 2 ** 18)
    ax[0].plot(w, db, color=INK, lw=0.7, ls="--",
               label=f"Butterworth + notch (Q {config.NOTCH_Q:g})")
    ax[0].set_xlim(config.MAINS_HZ - 10, config.MAINS_HZ + 10)
    ax[0].set_ylim(-80, 3)
    ax[0].set_xlabel("Frequency (Hz)")
    ax[0].set_ylabel("Zero-phase |H|² (dB)")
    ax[0].set_title(f"(a) {config.MAINS_HZ:g} Hz notch, zero-phase")
    ax[0].legend(frameon=False, fontsize=6.5, loc="lower left")
    jitter = np.random.default_rng(1)
    colours = (BLUE, ORANGE, GREEN)
    for j, i in enumerate(study.ids):
        for k, (name, colour) in enumerate(zip(STRESS_NAMES, colours)):
            gain = np.array(study.mc_raw[i][name]["gain"])
            x0 = j * 4 + k
            ax[1].scatter(x0 + jitter.uniform(-0.28, 0.28, gain.size), gain, s=1.5, color=colour,
                          alpha=0.35, lw=0)
            ax[1].errorbar(x0 + 0.42, gain.mean(), yerr=gain.std(ddof=1), fmt="o", ms=2.5,
                           color=INK, elinewidth=0.8, capsize=2)
    ax[1].set_xticks([j * 4 + 1 for j in range(len(study.ids))])
    ax[1].set_xticklabels([f"DS{i}" for i in study.ids])
    ax[1].set_ylabel("SNR gain (dB)")
    ax[1].set_title(f"(b) Stress test SNR gain, {study.n_draws} noise draws per record")
    handles = [Patch(color=c, alpha=0.7, label=n) for n, c in zip(STRESS_NAMES, colours)]
    handles.append(plt.Line2D([], [], color=INK, marker="o", ms=2.5, lw=0.8, label="mean ± SD"))
    ax[1].legend(handles=handles, frameon=False, fontsize=6.5, loc="upper center", ncol=4,
                 columnspacing=0.8, handlelength=1.2)
    ax[1].set_ylim(5, 36)
    _save(fig, out_dir, "fig7_notch_mc.png")


def filter_families(study, results, out_dir):
    """Appendix figure: each family at its minimum order for the zero-phase specification."""
    orders = results["min_orders_40_50_zero_phase"]
    names = {"butter": "Butterworth", "cheby1": "Chebyshev I", "cheby2": "Chebyshev II",
             "ellip": "Elliptic"}
    fig, ax = plt.subplots(1, 1, figsize=(5, 2.4))
    for (family, label), colour in zip(names.items(), SERIES):
        sos = flt.minimum_order_lowpass(family, orders[family], study.fs)
        w, db = _zp_db(sos, study.fs)
        ax.plot(w, db, color=colour, label=f"{label} (N={orders[family]})")
    ax.set_xlim(0, 100)
    ax.set_ylim(-100, 5)
    ax.axvspan(config.SPEC_PASS_HZ, config.SPEC_STOP_HZ, color=GREY, alpha=0.15, lw=0)
    ax.legend(frameon=False, fontsize=6.5)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Zero-phase |H|² (dB)")
    ax.set_title(f"Minimum-order low-pass, zero-phase: ≤{config.SPEC_PASS_DB:g} dB at "
                 f"{config.SPEC_PASS_HZ:g} Hz, ≥{config.SPEC_STOP_DB:g} dB at "
                 f"{config.SPEC_STOP_HZ:g} Hz")
    _save(fig, out_dir, "figA_families.png", dpi=250)


def make_all(study, results, out_dir):
    """Draw every figure into out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(STYLE)
    time_and_spectra(study, out_dir)
    aliasing(study, results, out_dir)
    convolution(study, out_dir)
    time_frequency(study, out_dir)
    dwt(study, results, out_dir)
    filter_parameters(study, out_dir)
    filter_performance(study, out_dir)
    notch_and_stress(study, out_dir)
    filter_families(study, results, out_dir)
