An MPI plugin for PySCF
=======================

mpi4pyscf is a plugin for PySCF which enables MPI (Message Passing Interface) parallelism.

Requires Python >= 3.13, pyscf >= 2.5 and mpi4py >= 4.0.

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
pip install . --no-deps
```
Recommended to clone repo and install from source with `--no-deps` flag to keep pip from replacing an existing `mpi4py` with a generic package. This is especially important if you have built `mpi4py` against system MPI libraries.

