"""Defect tests for product STATUS-layer reads (F14) feeding coverage, via `tags.py`."""

from pathlib import Path

import pandas as pd
import pytest
import requests
from click.testing import CliRunner
from rasterio.errors import RasterioIOError

from dist_s1_prod_check import tags
from dist_s1_prod_check.cli import cli
from factories import bursts_for_tile, dist_id, make_dist, make_rtc, rtc_id, seed_data_dir


TILE = '10SGC'
ACQ_READABLE = pd.Timestamp('2026-01-02T02:06:53')
ACQ_REPROCESSED = pd.Timestamp('2026-01-14T02:06:53')


def _group_bursts(tile: str, group: int) -> list[str]:
    lut = bursts_for_tile(tile)
    return sorted(lut[lut.acq_group_id_within_mgrs_tile == group].jpl_burst_id)


def _fake_get_tags(unreadable_urls: set[str], post_ids: dict[str, list[str]]) -> object:
    """STATUS-layer reader: 404 for `unreadable_urls`, otherwise tags naming the product's post inputs."""

    def get_tags(url: str) -> dict[str, str]:
        if url in unreadable_urls:
            raise RasterioIOError(f'{url}: HTTP response code: 404')
        opera_id = url.rsplit('/', 1)[-1].removesuffix('_GEN-DIST-STATUS.tif')
        return {
            'mgrs_tile_id': TILE,
            'pre_rtc_opera_ids': '',
            'post_rtc_opera_ids': ','.join(post_ids.get(opera_id, [])),
            'prior_dist_s1_product': 'None',
        }

    get_tags.retry_with = lambda **kwargs: get_tags  # `_preflight` calls get_tags.retry_with(...)
    return get_tags


# aa2014f: coverage CSV has no status column; the pass is product_found=True, matched to the unreadable latest version
@pytest.mark.xfail(strict=True, reason='F14: a 404 on the latest STATUS layer still counts as delivered for coverage')
def test_f14_latest_status_404_is_unknown_in_coverage(
    tmp_data_dir: Path, tmp_path: Path, cli_runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _blocked(*args: object, **kwargs: object) -> None:
        raise AssertionError('network access attempted in test')

    monkeypatch.setattr(requests, 'get', _blocked)
    monkeypatch.setattr(requests.Session, 'request', _blocked)

    bursts = _group_bursts(TILE, 0)
    readable = dist_id(TILE, ACQ_READABLE, '2026-01-02T12:00:00')
    older = dist_id(TILE, ACQ_REPROCESSED, '2026-01-14T12:00:00')
    latest = dist_id(TILE, ACQ_REPROCESSED, '2026-02-01T12:00:00')
    df_dist = make_dist([{'opera_id': i} for i in (readable, older, latest)])

    rtc_rows = [
        {'burst': b, 'acq': acq + pd.Timedelta(seconds=10 * k), 'proc': acq + pd.Timedelta(hours=5)}
        for acq in (ACQ_READABLE, ACQ_REPROCESSED)
        for k, b in enumerate(bursts)
    ]
    df_rtc = make_rtc(rtc_rows)
    post_ids = {
        readable: [rtc_id(r['burst'], r['acq'], r['proc']) for r in rtc_rows if r['acq'].day == ACQ_READABLE.day],
        older: [rtc_id(r['burst'], r['acq'], r['proc']) for r in rtc_rows if r['acq'].day == ACQ_REPROCESSED.day],
    }
    seed_data_dir(tmp_data_dir, dist=df_dist, rtc=df_rtc, venue='OPS-POP1', target_tiles=[TILE])

    # The latest version's STATUS layer 404s; the older version of the same acquisition is readable.
    latest_url = df_dist.set_index('opera_id').alert_url[latest]
    monkeypatch.setattr(tags, 'get_tags', _fake_get_tags({latest_url}, post_ids))

    out_dir = tmp_path / 'out'
    args = ['run-all', '--venue', 'OPS-POP1', '--start', '2026-01-01', '--stop', '2026-02-01']
    args += ['--workers', '1', '--inputs-workers', '1', '--data-dir', str(tmp_data_dir), '--out-dir', str(out_dir)]
    result = cli_runner.invoke(cli, args)
    assert result.exception is None, result.output

    df_tags = pd.read_parquet(tmp_data_dir / 'dist_s1_tags.parquet').set_index('opera_id')
    assert '404' in df_tags.error[latest], 'fixture: the latest STATUS read should have failed'

    (cov_csv,) = out_dir.glob('coverage_2*.csv')
    df_cov = pd.read_csv(cov_csv, keep_default_na=False)
    row = df_cov[pd.to_datetime(df_cov.post_acq_dt).dt.day == ACQ_REPROCESSED.day].iloc[0].to_dict()

    assert row['matched_opera_id'] == latest, 'must stay matched to the latest version, never the older readable copy'
    assert row.get('status') == 'UNKNOWN', f'latest product unreadable but coverage row is {row}'
    assert 'latest product unreadable' in row['reason']
