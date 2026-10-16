from pathlib import Path
from typing import NoReturn

import boto3
import geopandas as gpd
import pandas as pd
import pytest
import requests
from click.testing import CliRunner

from dist_s1_prod_check import ids
from dist_s1_prod_check.cli import cli
from dist_s1_prod_check.constants import DIST_S1_PARQUET, RTC_S1_PARQUET, TAGS_PARQUET
from factories import (
    DIST_COLUMNS,
    RTC_COLUMNS,
    TAG_COLUMNS,
    bursts_for_tile,
    dist_id,
    make_dist,
    make_rtc,
    make_tags,
    rtc_id,
    seed_data_dir,
)


REAL_DIR = Path.home() / 'data/opera/dist-s1-prod-check/pop1_aoi'
TILE = '10SGC'
ACQ = '2026-01-02T02:06:53'
PROC = '2026-05-07T23:35:27'


def _sample_frames() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, pd.DataFrame]:
    bursts = bursts_for_tile(TILE)
    group = bursts[bursts.track_number == bursts.track_number.iloc[0]].jpl_burst_id.tolist()
    post = [rtc_id(b, ACQ, '2026-01-02T07:41:51') for b in group]
    pre = [rtc_id(b, '2025-01-02T02:06:53', '2025-01-02T07:41:51') for b in group]
    dist = make_dist(
        [
            {'tile': TILE, 'acq': ACQ, 'proc': PROC},
            {'tile': TILE, 'acq': ACQ, 'proc': '2026-05-08T00:00:00'},
            {'tile': 'T10SGD', 'acq': ACQ, 'proc': PROC, 'sensor': 'S1C'},
        ]
    )
    rtc = make_rtc([{'opera_id': i} for i in post + pre])
    tags = make_tags(
        [{'opera_id': o, 'pre_rtc_opera_ids': pre, 'post_rtc_opera_ids': post} for o in dist.opera_id]
        + [{'opera_id': dist_id('10SGE', ACQ, PROC), 'error': 'RasterioIOError: 404'}]
    )
    return dist, rtc, tags


def test_dist_id_round_trip() -> None:
    real = 'OPERA_L3_DIST-ALERT-S1_T10SGC_20260102T020653Z_20260507T233527Z_S1A_30_v1.0'
    assert dist_id('T10SGC', ACQ, PROC) == real
    assert dist_id(TILE, pd.Timestamp(ACQ, tz='UTC'), pd.Timestamp(PROC).to_pydatetime()) == real
    oid = dist_id(TILE, ACQ, PROC, sensor='S1C', version='v2.0')
    assert ids.get_mgrs_tile_id(oid) == TILE
    assert ids.get_acq_timestamp_naive(oid) == pd.Timestamp(ACQ)
    assert ids.get_acq_timestamp(oid) == pd.Timestamp(ACQ, tz='UTC')
    assert ids.get_processing_timestamp_naive(oid) == pd.Timestamp(PROC)
    assert ids.get_opera_id_trunc(oid) == 'OPERA_L3_DIST-ALERT-S1_T10SGC_20260102T020653Z'
    assert oid.split('_')[6] == 'S1C' and oid.endswith('_v2.0')


def test_rtc_id_round_trip() -> None:
    real = 'OPERA_L2_RTC-S1_T035-073219-IW3_20221107T020647Z_20250222T111325Z_S1A_30_v1.0'
    oid = rtc_id('T035-073219-IW3', '2022-11-07 02:06:47', pd.Timestamp('2025-02-22 11:13:25', tz='UTC'))
    assert oid == real
    assert ids.get_burst_id(oid) == 'T035-073219-IW3'
    assert ids.get_track_number(ids.get_burst_id(oid)) == 35
    assert ids.get_acq_timestamp(oid) == pd.Timestamp('2022-11-07 02:06:47', tz='UTC')
    assert ids.get_opera_id_from_url(f'https://x/{oid}/{oid}_VV.tif') == oid


def test_frames_are_consistent() -> None:
    dist, rtc, tags = _sample_frames()
    assert list(dist.columns) == DIST_COLUMNS and list(rtc.columns) == RTC_COLUMNS
    assert list(tags.columns) == TAG_COLUMNS
    assert dist.crs == 'EPSG:4326' and rtc.crs == 'EPSG:4326'
    assert dist.geometry.notna().all() and rtc.geometry.notna().all()
    assert dist.opera_dedup_id.nunique() == 2
    assert (rtc.track_number == 35).all() or rtc.track_number.nunique() == 1
    bad = tags[tags.error != '']
    assert len(bad) == 1 and (bad[['mgrs_tile_id', 'pre_rtc_opera_ids']] == '').all(axis=None)
    assert (tags[tags.error == ''].prior_dist_s1_product == 'None').all()
    nulled = make_dist([{'tile': TILE, 'acq': ACQ, 'proc': PROC, 'alert_url': None}])
    assert nulled.alert_url.isna().all() and nulled.metric_url.notna().all()


@pytest.mark.skipif(not REAL_DIR.exists(), reason=f'{REAL_DIR} absent')
def test_schemas_match_real_caches(tmp_path: Path) -> None:
    dist, rtc, tags = _sample_frames()
    seed_data_dir(tmp_path, dist, rtc, tags)
    pairs = [
        (gpd.read_parquet, DIST_S1_PARQUET, dist),
        (gpd.read_parquet, RTC_S1_PARQUET, rtc),
        (pd.read_parquet, TAGS_PARQUET, tags),
    ]
    for reader, name, frame in pairs:
        real = reader(REAL_DIR / name)
        seeded = reader(tmp_path / name)
        if name == TAGS_PARQUET:
            # The real caches predate source_url (F23, Phase 2), which seed_data_dir records like fetch_tags_table.
            assert seeded.source_url.dtype == real.opera_id.dtype
            seeded = seeded.drop(columns='source_url')
        assert list(frame.columns) == list(real.columns), name
        pd.testing.assert_series_equal(frame.dtypes, real.dtypes, check_names=False, obj=f'{name} in memory')
        pd.testing.assert_series_equal(seeded.dtypes, real.dtypes, check_names=False, obj=f'{name} round trip')


def test_check_duplicates_offline(
    tmp_path: Path, cli_runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_data_dir: Path
) -> None:
    def _no_network(*args: object, **kwargs: object) -> NoReturn:
        raise AssertionError('network access attempted')

    monkeypatch.setattr(requests.Session, 'request', _no_network)
    monkeypatch.setattr(boto3, 'client', _no_network)
    dist, rtc, tags = _sample_frames()
    seed_data_dir(tmp_data_dir, dist, rtc, tags)
    out_dir = tmp_path / 'out'
    result = cli_runner.invoke(cli, ['check-duplicates', '--data-dir', str(tmp_data_dir), '--out-dir', str(out_dir)])
    assert result.exit_code == 0, result.output + repr(result.exception)
    csvs = list(out_dir.glob('*.csv'))
    assert csvs, f'no CSV written in {out_dir}'
    df = pd.concat(pd.read_csv(p) for p in csvs if p.stat().st_size > 1)
    assert len(df) == 2 and df.is_latest.sum() == 1
    assert list(out_dir.glob('report_*.html'))
