from pathlib import Path

import pandas as pd


def get_opera_id_trunc(opera_id: str) -> str:
    """Return an OPERA id truncated to product, tile/burst and acquisition time (drops processing time onward)."""
    return '_'.join(Path(opera_id).name.split('_')[:5])


def get_burst_id(opera_rtc_id: str) -> str:
    """Return the JPL burst id of an RTC-S1 OPERA id (or URL)."""
    return Path(opera_rtc_id).name.split('_')[3]


def get_track_number(jpl_burst_id: str) -> int:
    """Return the relative orbit (track) number of a JPL burst id such as ``T035-073226-IW3``."""
    return int(jpl_burst_id.split('-')[0][1:])


def get_acq_timestamp(opera_id: str) -> pd.Timestamp:
    """Return the acquisition time of an OPERA id as a UTC timestamp."""
    return pd.Timestamp(Path(opera_id).name.split('_')[4].rstrip('Z')).tz_localize('UTC')


def get_acq_timestamp_naive(opera_id: str) -> pd.Timestamp:
    """Return the acquisition time of an OPERA id as a timestamp without offset (UTC)."""
    return pd.Timestamp(Path(opera_id).name.split('_')[4].rstrip('Z'))


def get_processing_timestamp_naive(opera_id: str) -> pd.Timestamp:
    """Return the processing time of an OPERA id as a timestamp without offset (UTC)."""
    return pd.Timestamp(Path(opera_id).name.split('_')[5].rstrip('Z'))


def get_mgrs_tile_id(opera_dist_s1_id: str) -> str:
    """Return the MGRS tile id of a DIST-S1 OPERA id, without the leading ``T``."""
    return Path(opera_dist_s1_id).name.split('_')[3].lstrip('T')


def get_opera_id_from_url(url: str) -> str:
    """Return the OPERA id of a product URL (its parent directory name)."""
    return Path(url).parent.name
