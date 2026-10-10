"""Re-export of ``kernel_analyzer.contract_v3`` (the module moved into the package so that an installed wheel does not
depend on this repository path).  Kept for the scripts and tests that import ``contract_v3`` from here."""
import sys
from pathlib import Path

try:
    from kernel_analyzer.contract_v3 import *  # noqa: F401,F403
    from kernel_analyzer.contract_v3 import _verdict  # noqa: F401
except ImportError:  # a script run from the repository without the package on the path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from kernel_analyzer.contract_v3 import *  # noqa: F401,F403
    from kernel_analyzer.contract_v3 import _verdict  # noqa: F401
