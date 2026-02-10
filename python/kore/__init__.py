# kore — Pure Rust ML framework with Python bindings
# The native extension is loaded by maturin automatically.
from .kore import *  # noqa: F401,F403
from .kore import Tensor, nn, optim, functional  # noqa: F401
from .kore import save_state_dict, load_state_dict  # noqa: F401

__version__ = "0.1.0"
