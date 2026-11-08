"""Compatibility for checkpoint constructors under Transformers 5 meta init."""

import importlib


class _TorchConstants:
    def __init__(self, torch_module):
        self._torch = torch_module

    def __getattr__(self, name):
        return getattr(self._torch, name)

    def linspace(self, *args, **kwargs):
        # InternVisionEncoder converts the drop-path schedule to Python floats.
        # This constant must be materialized even when weights use meta init.
        kwargs.setdefault('device', 'cpu')
        return self._torch.linspace(*args, **kwargs)


def prepare_checkpoint_class(path: str) -> None:
    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    reference = config.auto_map['AutoModel']
    cls = get_class_from_dynamic_module(reference, path)
    package = cls.__module__.rsplit('.', 1)[0]
    vision_module = importlib.import_module(f'{package}.modeling_intern_vit')
    if not isinstance(vision_module.torch, _TorchConstants):
        vision_module.torch = _TorchConstants(vision_module.torch)
    if 'all_tied_weights_keys' not in cls.__dict__:
        cls.all_tied_weights_keys = {}
