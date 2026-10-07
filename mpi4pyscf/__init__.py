'''
An MPI plugin for PySCF
'''

__version__ = '0.4.0'

import re
import pyscf

def _version_tuple(v):
    return tuple(int(x) for x in re.findall(r'\d+', v)[:3])

if _version_tuple(pyscf.__version__) < (2,5,0):
    raise ImportError('mpi4pyscf %s requires pyscf>=2.5.0, found %s' 
                        %(__version__, pyscf.__version__)
                    )

del _version_tuple, re

# import all pyscf submodules before suspending the slave processes
from pyscf import __all__

# NOTE: suspend all slave processes at last
from .tools import mpi
if not mpi.pool.is_master():
    import sys
    import traceback
    
    try:
        sys.modules[__name__].__spec__._initializing = False
    except AttributeError:
        pass

    try:
        mpi.pool.wait()
    except BaseException as err:
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        mpi.comm.Abort(1)
        exit(1)

    # Ensure mpi processes terminated
    exit(0)
