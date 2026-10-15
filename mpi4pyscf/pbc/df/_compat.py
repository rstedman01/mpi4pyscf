'''
Adapters between the PySCF 1.x density-fitting internals that mpi4pyscf.pbc.df
was written against and the builder-based implementation of PySCF >= 2.x
(pyscf.pbc.df.gdf_builder).
'''

import os
import re
import numpy
import pyscf
from pyscf.pbc.df import aft
from pyscf.pbc.df import gdf_builder
from pyscf.pbc.gto import pseudo

PYSCF_VERSION = tuple(int(x) for x in re.findall(r'\d+', pyscf.__version__)[:3])

# PySCF 1.5 - 2.12 evaluate the G=0 term of the exact exchange analytically
# (df_jk._ewald_exxdiv_for_G0) after the k-loop and build coulG without it.
# PySCF 2.13 instead puts the G=0 term into coulG, i.e.
# get_coulG(cell, k, exxdiv, ...), and no longer calls _ewald_exxdiv_for_G0.
# The two differ by FFT discretization errors, so follow the installed PySCF
# to stay comparable with its serial results.
EXXDIV_IN_COULG = PYSCF_VERSION >= (2, 13, 0)


def fuse_auxcell(mydf, auxcell):
    '''PySCF 1.x signature: fuse_auxcell(mydf, auxcell) -> (fused_cell, fuse).

    PySCF 2.x moved the function to gdf_builder and replaced the density-fitting
    object by the smoothing exponent: fuse_auxcell(auxcell, eta).  In 2.x
    GDF.eta stays None until the builder estimates it, so estimate it here in
    the same way when it has not been set.
    '''
    eta = getattr(mydf, 'eta', None)
    if eta is None:
        eta = gdf_builder._guess_eta(auxcell, mydf.kpts, mydf.mesh)[0]
    return gdf_builder.fuse_auxcell(auxcell, eta)


def unpack_(mydf, dfdic):
    '''Restore the attributes sent by DF.pack() on the object.

    In PySCF 2.x several of them (kpts, ...) are properties backed by private
    attributes, so updating __dict__ directly would be silently ignored.  Use
    setattr, but leave kpts alone when it is unchanged: pack() sends absolute
    k-points, and setting them would replace e.g. a KPoints object (with
    symmetry) held by the object on the master process by a plain array.
    '''
    for key, val in dfdic.items():
        if key == 'kpts':
            old = mydf.kpts
            if (numpy.shape(old) == numpy.shape(val) and
                    numpy.allclose(old, val, rtol=0, atol=1e-12)):
                continue
        setattr(mydf, key, val)
    return mydf


def check_kpts(mydf, kpts):
    '''Returns (kpts as an (nk,3) array, whether a single k-point was given).

    Since PySCF 2.10 get_nuc/get_pp return an (nk,nao,nao) array whenever kpts
    is None, taking mydf.kpts; before they returned (nao,nao) for kpts=None.
    '''
    if PYSCF_VERSION >= (2, 10, 0):
        return aft._check_kpts(mydf, kpts)
    if kpts is None:
        return numpy.zeros((1,3)), True
    return numpy.reshape(kpts, (-1,3)), numpy.shape(kpts) == (3,)


def get_pp_loc_part1(mydf, kpts, with_pseudo=True):
    '''Serial PW-expanded local pseudopotential/nuclear attraction.  The work
    is divided over the MPI ranks by AFTDF.prange, and the caller reduces.'''
    if hasattr(aft, '_get_pp_loc_part1'):
        return aft._get_pp_loc_part1(mydf, kpts, with_pseudo=with_pseudo)
    return aft.get_pp_loc_part1(mydf, kpts)


def get_pp_loc_part2_and_nl(cell, kpts):
    '''Short-range local part and non-local part of the pseudopotential.'''
    if hasattr(aft, '_IntPPBuilder'):
        vloc2 = aft._IntPPBuilder(cell, kpts).get_pp_loc_part2()
    else:
        vloc2 = pseudo.pp_int.get_pp_loc_part2(cell, kpts)
    return numpy.asarray(vloc2) + numpy.asarray(pseudo.pp_int.get_pp_nl(cell, kpts))


def auxbar(mydf, fused_cell):
    '''PySCF 1.x: mydf.auxbar(fused_cell); PySCF 2.x: gdf_builder.auxbar.'''
    if hasattr(mydf, 'auxbar'):
        return mydf.auxbar(fused_cell)
    return gdf_builder.auxbar(fused_cell)


def require_experimental(name):
    '''Guard for the density-fitting classes (GDF/MDF) that are not ported yet.

    Their parallel integral build (_make_j3c) is a copy of the PySCF 1.7
    implementation and the class hierarchy no longer matches PySCF 2.x (GDF does
    not derive from AFTDF any more), so the results are not validated.  Fail
    loudly instead of returning plausible but unchecked numbers.  Set the
    environment variable MPI4PYSCF_EXPERIMENTAL_PBC_DF=1 to use them anyway
    (e.g. to continue the port).
    '''
    if os.environ.get('MPI4PYSCF_EXPERIMENTAL_PBC_DF') != '1':
        raise NotImplementedError(
            'mpi4pyscf.pbc.df.%s has not been ported to PySCF >= 2.x.  Use '
            'FFTDF or AFTDF, or set MPI4PYSCF_EXPERIMENTAL_PBC_DF=1 to use it '
            'without validation.' % name)
