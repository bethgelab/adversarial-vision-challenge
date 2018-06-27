from os.path import join as _join
from os.path import dirname as _dirname

with open(_join(_dirname(__file__), 'VERSION')) as _f:
    __version__ = _f.read().strip()

from .server import model_server  # noqa: F401
from .client import TinyImageNetBSONModel  # noqa: F401
from .utils import load_model  # noqa: F401
from .utils import read_images  # noqa: F401
from .utils import store_adversarial  # noqa: F401
from .utils import get_test_data  # noqa: F401
from .utils import attack_complete # noqa: F401
from .notifier import ModelNotifications, AttackNotifications # noqa: F401
