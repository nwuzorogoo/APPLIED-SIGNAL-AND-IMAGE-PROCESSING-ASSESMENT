"""CMP9780 A01: EEG signal analysis and digital filter design.

Modules
    config       analysis settings (bands, filter specifications, windows, Monte Carlo)
    edf_io       EDF loading and header calibration
    spectral     custom radix-2 FFT, amplitude spectrum, band power, peak picking
    convolution  direct convolution and smoothing kernels
    timefreq     STFT, Morlet CWT, DWT and operation counts
    filters      band-pass and notch design, zero-phase filtering, mains detector
    events       pulse and theta-burst detection, windows relative to events
    evaluation   stress-test contamination, metrics and paired statistics
    analysis     the per-task analysis that produces results.json
    plotting     report figures
    cli          command-line entry point
"""
__version__ = "1.0.0"
