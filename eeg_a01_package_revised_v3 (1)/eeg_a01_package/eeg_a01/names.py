"""Labels shared by the analysis, the benchmarks and the plots."""

FILTER_NAMES = {"butter": "Butterworth (N=4)", "ellip": "Elliptic (N=4)",
                "butter_notch": "Butterworth + notch"}
STRESS_NAMES = ("Butterworth", "Elliptic", "Butterworth+notch")


def freq_key(freq):
    """Short key for a frequency, such as 29.39 -> '29p4'."""
    return f"{freq:.1f}".replace(".", "p")
