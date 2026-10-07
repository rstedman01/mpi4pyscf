An MPI plugin for PySCF
=======================

mpi4pyscf is a plugin for PySCF which enables MPI (Message Passing Interface) parallelism.

Requires Python>=3.13, pyscf>=2.5 and mpi4py>=4.0 (the development branch is
tested with pyscf 2.13, mpi4py 4.1 and Open MPI 4.1).

Quick start
-----------

When the script of the serial PySCF version works, MPI can be activated by
importing mpi4pyscf and replacing the corresponding initialization statements in
the script. For example:
```
import pyscf
mol = pyscf.M(atom='''
O    0.   0.       0.
H    0.   -0.757   0.587
H    0.   0.757    0.587''',
               basis='cc-pvtz')

# Serial mode
from pyscf import scf
mf = scf.RHF(mol).run()

# MPI parallelism
from mpi4pyscf import scf
mf = scf.RHF(mol).run()
```

See more examples of usage in [examples](https://github.com/pyscf/mpi4pyscf/tree/master/examples).

Installation
------------

```
pip install --no-deps .
```
`--no-deps` keeps pip from replacing an existing mpi4py that was built against
the system MPI library (e.g. on a cluster) with a generic one.

Parallel CCSD(T)
----------------

```
from pyscf import gto, scf
from mpi4pyscf import cc

mol = gto.M(atom='...', basis='cc-pvtz')
mf = scf.RHF(mol).run()
mycc = cc.RCCSD(mf, frozen=1)
mycc.kernel()
et = mycc.ccsd_t()
```
run as `mpiexec -np N python script.py`.  Every rank executes the script, but
all ranks except rank 0 are suspended inside `import mpi4pyscf` and only follow
the work sent by rank 0.  Notes:

* The MPI library must provide `MPI_THREAD_MULTIPLE` (a warning is printed
  otherwise).
* Import `mpi4pyscf` before doing any real work: everything that runs before the
  import is executed by every rank.
* Set `OMP_NUM_THREADS` to the number of cores per MPI rank.
* To restart CCSD, set `mycc.diis_file` and later call
  `mycc.restore_from_diis_(diis_file)`.  The DIIS files are per rank
  (`<diis_file>__rank<i>`), so restart with the same number of ranks.  Use a
  different `diis_file` for the continuing run.
* Tests: `mpiexec -np 2 tests/runtests.sh`.

Not supported
-------------

The periodic modules `mpi4pyscf.pbc.df` (GDF/MDF/AFT) rely on PySCF internals
that were rewritten in PySCF 2.x and no longer import.
