"""Loading the EDF recordings and reading their header calibration."""
from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np

from . import config

_FIXED_HEADER_BYTES = 256
# Per-signal header fields in file order, with their width in bytes (EDF specification)
_SIGNAL_FIELDS = (
    ("label", 16), ("transducer", 80), ("physical_dimension", 8),
    ("physical_min", 8), ("physical_max", 8), ("digital_min", 8), ("digital_max", 8),
    ("prefiltering", 80), ("samples_per_record", 8), ("reserved", 32),
)


@dataclass(frozen=True)
class ChannelCalibration:
    """Physical and digital range of one EDF channel."""
    label: str
    unit: str
    physical_min: float
    physical_max: float
    digital_min: int
    digital_max: int

    @property
    def lsb(self):
        """Quantisation step in physical units per digital count."""
        return (self.physical_max - self.physical_min) / (self.digital_max - self.digital_min)

    @property
    def full_scale(self):
        """Largest representable magnitude in physical units."""
        return max(abs(self.physical_min), abs(self.physical_max))


@dataclass(frozen=True)
class Recording:
    """One single-channel EEG record in microvolts."""
    record_id: int
    data: np.ndarray
    fs: float
    calibration: ChannelCalibration

    @property
    def n_samples(self):
        return self.data.size

    @property
    def duration(self):
        return self.data.size / self.fs

    @property
    def time(self):
        return np.arange(self.data.size) / self.fs


def read_calibration(path, channel=0):
    """Read the physical and digital range of one channel straight from the EDF header."""
    raw = Path(path).read_bytes()
    if len(raw) < _FIXED_HEADER_BYTES:
        raise ValueError(f"{path} is too short to be an EDF file")
    n_signals = int(raw[252:256].decode("ascii").strip())
    if not 0 <= channel < n_signals:
        raise ValueError(f"channel {channel} not in file with {n_signals} signals")
    fields, offset = {}, _FIXED_HEADER_BYTES
    for name, width in _SIGNAL_FIELDS:
        start = offset + channel * width
        fields[name] = raw[start:start + width].decode("ascii").strip()
        offset += n_signals * width
    return ChannelCalibration(
        label=fields["label"], unit=fields["physical_dimension"],
        physical_min=float(fields["physical_min"]), physical_max=float(fields["physical_max"]),
        digital_min=int(fields["digital_min"]), digital_max=int(fields["digital_max"]),
    )


def load_recording(path, record_id):
    """Load the first EEG channel of an EDF file, converted from volts to microvolts."""
    raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
    calibration = read_calibration(path)
    if calibration.unit not in ("uV", "µV"):
        raise ValueError(f"{path}: expected a microvolt channel, header says {calibration.unit!r}")
    data = raw.get_data()[0] * 1e6
    return Recording(record_id, data, float(raw.info["sfreq"]), calibration)


def load_recordings(data_dir, record_ids=config.RECORD_IDS, pattern=config.EDF_PATTERN):
    """Load all records and check they share one sampling rate and length."""
    data_dir = Path(data_dir)
    records = {}
    for record_id in record_ids:
        path = data_dir / pattern.format(record_id)
        if not path.is_file():
            raise FileNotFoundError(f"missing recording {path}")
        records[record_id] = load_recording(path, record_id)
    rates = {r.fs for r in records.values()}
    lengths = {r.n_samples for r in records.values()}
    if len(rates) != 1 or len(lengths) != 1:
        raise ValueError(f"records differ in sampling rate {rates} or length {lengths}")
    return records
