import os
from .registry import model_entrypoint

if os.getenv('CURRENT_TASK') == 'VIS':
    from . import VIS
else:
    raise ValueError()









