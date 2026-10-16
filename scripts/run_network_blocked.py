"""Run the dist-s1-prod-check CLI with every network path patched to raise.

Usage::

    run_network_blocked.py <dist-s1-prod-check args...>
    run_network_blocked.py --repro-baseline NAME [--repro-tag TAG] [--drop-inputs-checkpoint] [--adopt] [extra args...]

The first form passes the arguments straight to ``dist_s1_prod_check.cli.cli``. The second form reproduces a
baseline from ``docs/prod-gate-hardening/baselines.json``: it verifies the frozen cache against ``SHA256SUMS``,
copies it to a fresh scratch directory ``<repro_root>/<NAME>[/<TAG>]/data`` (run-all writes into ``--data-dir``, and
the frozen copies are read-only), empties ``.../out``, and runs the baseline's ``original_args`` with ``--data-dir``
and ``--out-dir`` pointed there, followed by any extra args. ``--drop-inputs-checkpoint`` deletes
``inputs_check_results.parquet`` from the scratch copy so the inputs check is recomputed instead of replayed.
``--adopt`` (needed since Phase 1, which refuses caches without a manifest) first runs ``adopt-cache`` on the scratch
copy with the window, AOI and venue of ``original_args``, then appends ``--offline --regression-baseline``.

Blocked before the CLI is imported: ``requests.Session.request``, ``urllib3`` connection pools (requests and
botocore), ``boto3.client``/``boto3.resource`` and their ``Session`` methods, ``botocore`` HTTP sends,
``urllib.request`` openers, ``rasterio.open`` (GDAL/libcurl tag reads), ``earthaccess.login``, the ``asf_search``
search entry points, and ``socket`` connects and name lookups for non-loopback hosts. HTTP(S) proxy variables are
also pointed at a closed local port so that any libcurl access outside ``rasterio.open`` fails too.

The patches are installed at import time, so the worker processes that the inputs check spawns (which re-import
this module as ``__mp_main__``) are blocked too. Several call sites catch every exception (e.g. tag reads record
the error and move on), so each blocked attempt is also appended to the file named by
``DIST_S1_NETWORK_BLOCK_LOG``. The exit code is 0 for a clean run, 1 if the CLI raised, and 2 if any network access
was attempted, even if the CLI swallowed it.
"""

import argparse
import hashlib
import json
import os
import shutil
import socket
import stat
import sys
import tempfile
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any


BLOCK_LOG_ENV = 'DIST_S1_NETWORK_BLOCK_LOG'
DEFAULT_BASELINES = Path(__file__).resolve().parents[2] / 'docs' / 'prod-gate-hardening' / 'baselines.json'
LOOPBACK = {None, '', 'localhost', '127.0.0.1', '::1'}
DEAD_PROXY = 'http://127.0.0.1:9'
INPUTS_CHECKPOINT = 'inputs_check_results.parquet'


class NetworkBlockedError(RuntimeError):
    """Raised by every patched network entry point."""


def _record(target: str, detail: str) -> None:
    line = f'{os.getpid()}\t{target}\t{detail[:300]}\n'
    sys.stderr.write(f'NETWORK BLOCKED: {line}')
    log_path = os.environ.get(BLOCK_LOG_ENV)
    if log_path:
        with Path(log_path).open('a') as f:
            f.write(line)


def _blocker(target: str) -> Callable[..., Any]:
    def _raise(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        shown = [a for a in args[:3] if not hasattr(a, '__dict__')] + [kwargs[k] for k in ('url', 'fp') if k in kwargs]
        detail = ' '.join(repr(a) for a in shown)
        _record(target, detail)
        raise NetworkBlockedError(f'network access blocked: {target}({detail})')

    return _raise


def _is_remote(address: Any) -> bool:  # noqa: ANN401
    host = address[0] if isinstance(address, tuple) and address else address
    if isinstance(host, bytes):
        host = host.decode()
    return host not in LOOPBACK


def _patch_socket() -> None:
    real_connect, real_connect_ex = socket.socket.connect, socket.socket.connect_ex
    real_getaddrinfo, real_create_connection = socket.getaddrinfo, socket.create_connection

    def connect(self: socket.socket, address: Any) -> Any:  # noqa: ANN401
        if self.family in (socket.AF_INET, socket.AF_INET6) and _is_remote(address):
            _blocker('socket.connect')(address)
        return real_connect(self, address)

    def connect_ex(self: socket.socket, address: Any) -> Any:  # noqa: ANN401
        if self.family in (socket.AF_INET, socket.AF_INET6) and _is_remote(address):
            _blocker('socket.connect_ex')(address)
        return real_connect_ex(self, address)

    def getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        if _is_remote(host):
            _blocker('socket.getaddrinfo')(host)
        return real_getaddrinfo(host, *args, **kwargs)

    def create_connection(address: Any, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        if _is_remote(address):
            _blocker('socket.create_connection')(address)
        return real_create_connection(address, *args, **kwargs)

    socket.socket.connect = connect
    socket.socket.connect_ex = connect_ex
    socket.getaddrinfo = getaddrinfo
    socket.create_connection = create_connection


def install_network_blocks() -> list[str]:
    """Patch every known network entry point to raise ``NetworkBlockedError``; return what was patched."""
    for var in ('http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'GDAL_HTTP_PROXY'):
        os.environ[var] = DEAD_PROXY
    for var in ('no_proxy', 'NO_PROXY'):
        os.environ.pop(var, None)
    os.environ['AWS_EC2_METADATA_DISABLED'] = 'true'

    patched = []

    def patch(module_name: str, attr_path: str) -> None:
        try:
            obj = __import__(module_name, fromlist=['_'])
        except ImportError:
            return
        *parents, name = attr_path.split('.')
        for parent in parents:
            obj = getattr(obj, parent, None)
            if obj is None:
                return
        if hasattr(obj, name):
            setattr(obj, name, _blocker(f'{module_name}.{attr_path}'))
            patched.append(f'{module_name}.{attr_path}')

    patch('requests', 'Session.request')
    patch('urllib3.connectionpool', 'HTTPConnectionPool.urlopen')
    patch('boto3', 'client')
    patch('boto3', 'resource')
    patch('boto3.session', 'Session.client')
    patch('boto3.session', 'Session.resource')
    patch('botocore.httpsession', 'URLLib3Session.send')
    patch('urllib.request', 'OpenerDirector.open')
    patch('rasterio', 'open')
    patch('earthaccess', 'login')
    for fn in ('search', 'geo_search', 'granule_search', 'product_search', 'search_generator'):
        patch('asf_search', fn)
    _patch_socket()
    patched.append('socket.{connect,connect_ex,getaddrinfo,create_connection} (non-loopback)')
    return patched


if BLOCK_LOG_ENV not in os.environ:
    fd, _log = tempfile.mkstemp(prefix='network_block_', suffix='.log')
    os.close(fd)
    os.environ[BLOCK_LOG_ENV] = _log
PATCHED = install_network_blocks()


def _verify_frozen(frozen_root: Path, name: str) -> None:
    sums = (frozen_root / 'SHA256SUMS').read_text().splitlines()
    entries = [line.split(maxsplit=1) for line in sums if line.split(maxsplit=1)[1].startswith(f'{name}/')]
    if not entries:
        raise SystemExit(f'No SHA256SUMS entries for {name}')
    for digest, rel in entries:
        if hashlib.sha256((frozen_root / rel).read_bytes()).hexdigest() != digest:
            raise SystemExit(f'Frozen cache changed: {rel} no longer matches SHA256SUMS')


def _make_writable(root: Path) -> None:
    for path in [root, *root.rglob('*')]:
        path.chmod(path.stat().st_mode | stat.S_IWUSR)


def prepare_repro(name: str, tag: str | None, drop_inputs_checkpoint: bool, baselines_path: Path) -> list[str]:
    """Create a scratch copy of a frozen baseline cache and return the CLI args that reproduce the baseline."""
    baselines = json.loads(baselines_path.read_text())
    entry = baselines['baselines'][name]
    root = Path(baselines['data_root']).expanduser()
    _verify_frozen(root / baselines['frozen_root'], name)
    work = root / baselines['repro_root'] / name
    work = work / tag if tag else work
    data_dir, out_dir = work / 'data', work / 'out'
    for path in (data_dir, out_dir):
        if path.exists():
            _make_writable(path)
            shutil.rmtree(path)
    shutil.copytree(root / entry['frozen_cache'], data_dir)
    _make_writable(data_dir)
    if drop_inputs_checkpoint:
        (data_dir / INPUTS_CHECKPOINT).unlink(missing_ok=True)

    args = list(entry['original_args'])
    for flag, value in (('--data-dir', data_dir), ('--out-dir', out_dir)):
        if flag in args:
            args[args.index(flag) + 1] = str(value)
        else:
            args += [flag, str(value)]
    return args


ADOPT_FLAGS = {'--start': 1, '--stop': 1, '--bbox': 4, '--venue': 1, '--data-dir': 1, '--tag-url-template': 1}


def adopt_args(run_args: list[str]) -> list[str]:
    """Return the ``adopt-cache`` arguments (window, AOI, venue, data dir) of a ``run-all`` command line."""
    out, i = ['adopt-cache'], 0
    while i < len(run_args):
        n = ADOPT_FLAGS.get(run_args[i])
        if n is None:
            i += 1
            continue
        out += run_args[i : i + 1 + n]
        i += 1 + n
    return out


def main(argv: list[str]) -> int:
    """Run the CLI with the network blocked; return the process exit code."""
    if argv[:1] == ['--repro-baseline']:
        parser = argparse.ArgumentParser(description='Reproduce a baseline run with the network blocked.')
        parser.add_argument('--repro-baseline', required=True)
        parser.add_argument('--repro-tag', default=None)
        parser.add_argument('--drop-inputs-checkpoint', action='store_true')
        parser.add_argument('--adopt', action='store_true')
        parser.add_argument('--baselines', type=Path, default=DEFAULT_BASELINES)
        opts, extra = parser.parse_known_args(argv)
        argv = prepare_repro(opts.repro_baseline, opts.repro_tag, opts.drop_inputs_checkpoint, opts.baselines) + extra
        if opts.adopt:
            from dist_s1_prod_check.cli import cli

            adopt = adopt_args(argv)
            print(f'dist-s1-prod-check {" ".join(adopt)}', file=sys.stderr)
            cli.main(args=adopt, prog_name='dist-s1-prod-check', standalone_mode=False)
            argv += ['--offline', '--regression-baseline']

    print(f'network blocked: {", ".join(PATCHED)}', file=sys.stderr)
    print(f'dist-s1-prod-check {" ".join(argv)}', file=sys.stderr)
    from dist_s1_prod_check.cli import cli

    code = 0
    try:
        cli.main(args=argv, prog_name='dist-s1-prod-check', standalone_mode=False)
    except Exception:
        traceback.print_exc()
        code = 1
    attempts = Path(os.environ[BLOCK_LOG_ENV]).read_text()
    if attempts:
        print(f'{attempts.count(chr(10))} blocked network attempt(s):\n{attempts}', file=sys.stderr)
        return 2
    print('no network access attempted', file=sys.stderr)
    return code


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
