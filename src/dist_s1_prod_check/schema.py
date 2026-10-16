"""Schema validation of the cached DIST-S1 and RTC-S1 tables on load (F35).

A hand-edited or truncated cache must not reach the checks: a NaT processing time, for instance, compares as
neither earlier nor later and silently passes the ordering check. Every load validates the columns the checks rely
on (no nulls, unique ``opera_id``, expected dtypes) and refuses the table with every failing check listed.
"""

import click
import pandas as pd
import pandera.pandas as pa
from pandas.api.types import is_datetime64_any_dtype, is_integer_dtype, is_string_dtype


def _dtype(predicate: object, name: str) -> pa.Check:
    return pa.Check(lambda s: bool(predicate(s.dtype)), name=f'dtype is {name}', error=f'dtype is not {name}')


def _naive_datetime() -> pa.Check:
    return pa.Check(
        lambda s: is_datetime64_any_dtype(s.dtype) and getattr(s.dtype, 'tz', None) is None,
        name='dtype is datetime (UTC, no offset)',
        error='dtype is not a tz-naive datetime',
    )


def _utc_datetime() -> pa.Check:
    return pa.Check(
        lambda s: is_datetime64_any_dtype(s.dtype) and str(getattr(s.dtype, 'tz', None)) == 'UTC',
        name='dtype is datetime (UTC)',
        error='dtype is not a UTC datetime',
    )


def _string(nullable: bool = False, unique: bool = False) -> pa.Column:
    return pa.Column(None, nullable=nullable, unique=unique, checks=_dtype(is_string_dtype, 'string'))


DIST_SCHEMA = pa.DataFrameSchema(
    {
        'opera_id': _string(unique=True),
        'opera_dedup_id': _string(),
        'mgrs_tile_id': _string(),
        'acq_time': pa.Column(None, nullable=False, checks=_naive_datetime()),
        'processing_time': pa.Column(None, nullable=False, checks=_naive_datetime()),
        'alert_url': _string(nullable=True),
    },
    name='dist_s1_products',
)

RTC_SCHEMA = pa.DataFrameSchema(
    {
        'opera_id': _string(unique=True),
        'opera_dedup_id': _string(),
        'jpl_burst_id': _string(),
        'acq_dt': pa.Column(None, nullable=False, checks=_utc_datetime()),
        'processing_dt': pa.Column(None, nullable=False, checks=_utc_datetime()),
        'track_number': pa.Column(None, nullable=False, checks=_dtype(is_integer_dtype, 'integer')),
        'polarizations': _string(),
    },
    name='rtc_s1_products',
)


def _describe(errors: pa.errors.SchemaErrors) -> str:
    cases = errors.failure_cases
    parts = []
    for (column, check), group in cases.groupby(['column', 'check'], dropna=False, sort=True):
        values = [str(v) for v in group.failure_case.dropna().unique()[:3] if not isinstance(v, bool)]
        examples = ', '.join(values)
        parts.append(f'{column}: {check} ({len(group)} failing' + (f', e.g. {examples})' if examples else ')'))
    return '; '.join(parts)


def validate(df: pd.DataFrame, schema: pa.DataFrameSchema, path: object) -> pd.DataFrame:
    """Return `df` unchanged if it satisfies `schema`, else raise a `ClickException` listing every failure."""
    try:
        schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as e:
        raise click.ClickException(
            f'{path} fails schema validation: {_describe(e)}. Re-download it with --refresh.'
        ) from None
    return df
