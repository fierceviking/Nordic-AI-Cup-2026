"""Solution package for the medical-appointment case.

The Apple Silicon environment variables are set here rather than left to the
shell, because ``python api.py`` has to behave the same way the tools do. They
have to be in place before torch is first imported, and every torch import in
this package is lazy, so importing the package is early enough.
"""

import os

# A few ops the NLI reader uses have no Metal kernel; without this they raise
# rather than falling back to CPU.
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
