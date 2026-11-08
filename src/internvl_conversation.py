"""Resolve prompt templates from the checkpoint's own remote-code module."""

import importlib
import importlib.util
import os
import sys
from pathlib import Path

_MODULE = None


def register_model(model) -> None:
    """Keep the exact conversation implementation shipped with the model."""
    global _MODULE
    package = type(model).__module__.rsplit('.', 1)[0]
    try:
        _MODULE = importlib.import_module(f'{package}.conversation')
        return
    except ModuleNotFoundError:
        directory = os.environ.get('INTERNVL_MODEL_DIR')
        if not directory:
            raise RuntimeError(
                'Cannot resolve the checkpoint conversation module. Set '
                'INTERNVL_MODEL_DIR to its local snapshot directory.'
            ) from None
        path = Path(directory) / 'conversation.py'
        spec = importlib.util.spec_from_file_location('_twinprune_conversation', path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _MODULE = module


def get_conv_template(name):
    if _MODULE is None:
        raise RuntimeError('Load an InternVL model before requesting its template.')
    return _MODULE.get_conv_template(name)
