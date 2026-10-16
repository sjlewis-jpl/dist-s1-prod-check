import importlib

import pytest


MODULES = [
    'dist_s1_prod_check',
    'dist_s1_prod_check.artifacts',
    'dist_s1_prod_check.checks',
    'dist_s1_prod_check.cli',
    'dist_s1_prod_check.cmr',
    'dist_s1_prod_check.constants',
    'dist_s1_prod_check.coverage',
    'dist_s1_prod_check.ids',
    'dist_s1_prod_check.inputs',
    'dist_s1_prod_check.report',
    'dist_s1_prod_check.tags',
]


@pytest.mark.parametrize('module', MODULES)
def test_module_imports(module: str) -> None:
    importlib.import_module(module)
