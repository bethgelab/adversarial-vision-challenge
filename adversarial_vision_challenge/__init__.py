from os.path import join as _join
from os.path import dirname as _dirname

with open(_join(_dirname(__file__), 'VERSION')) as _f:
    __version__ = _f.read().strip()

from .server import mnist_model_server  # noqa: F401
from .server import cifar_model_server  # noqa: F401
from .server import imagenet_model_server  # noqa: F401
from .server import attack_server  # noqa: F401
from .client import BSONModel  # noqa: F401
from .client import BSONAttack  # noqa: F401
