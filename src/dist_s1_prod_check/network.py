"""Offline mode: one switch, checked at every point where the tool would reach the network.

`network_allowed(step)` is called at the top of `cmr.search_granules`, `cmr._list_s3_prefixes`, `tags.get_tags`,
`tags.fetch_tags_table` and `inputs._enumerate_one`. In offline mode it raises an `OfflineError` naming the step,
so a run can only use cached data and never fetches or re-queries anything silently.
"""

import click


_offline = False


class OfflineError(click.ClickException):
    """A step needed the network while --offline was set."""


def set_offline(offline: bool) -> None:
    global _offline  # noqa: PLW0603
    _offline = bool(offline)


def is_offline() -> bool:
    return _offline


def network_allowed(step: str) -> None:
    """Raise `OfflineError` if offline mode is on; `step` names what needed the network."""
    if _offline:
        raise OfflineError(
            f'--offline is set but {step} needs network access; run without --offline or seed the cache.'
        )
