from pathlib import Path

import pandas as pd


def get_opera_id_trunc(opera_id: str) -> str:
    return '_'.join(Path(opera_id).name.split('_')[:5])


def get_burst_id(opera_rtc_id: str) -> str:
    return Path(opera_rtc_id).name.split('_')[3]


def get_track_number(jpl_burst_id: str) -> int:
    return int(jpl_burst_id.split('-')[0][1:])


def get_acq_timestamp(opera_id: str) -> pd.Timestamp:
    return pd.Timestamp(Path(opera_id).name.split('_')[4].rstrip('Z')).tz_localize('UTC')


def get_acq_timestamp_naive(opera_id: str) -> pd.Timestamp:
    return pd.Timestamp(Path(opera_id).name.split('_')[4].rstrip('Z'))


def get_processing_timestamp_naive(opera_id: str) -> pd.Timestamp:
    return pd.Timestamp(Path(opera_id).name.split('_')[5].rstrip('Z'))


def get_mgrs_tile_id(opera_dist_s1_id: str) -> str:
    return Path(opera_dist_s1_id).name.split('_')[3].lstrip('T')


def get_opera_id_from_url(url: str) -> str:
    return Path(url).parent.name
