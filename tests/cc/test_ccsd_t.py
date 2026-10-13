#!/usr/bin/env python

'''
Parallel RCCSD and RCCSD(T) against the serial PySCF implementation.

    mpiexec -np 2 tests/runtests.sh
'''

import pytest
from numpy.testing import assert_allclose
from pyscf import gto, scf, cc
from mpi4pyscf import cc as mpicc


@pytest.fixture(scope='module')
def get_mf():
    mol = gto.M(atom='O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587',
                basis='cc-pvdz', verbose=0)
    # A tight SCF matters for (T): the occupied-virtual Fock block (the SCF
    # orbital gradient) enters the triples energy, and PySCF's MPI (T) kernel
    # treats that term slightly differently from the serial kernel.  With the
    # default SCF settings this alone shifts E(T) by ~1e-10 Eh.
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-9
    return mf.run()


@pytest.mark.parametrize('frozen', [0, 1])
def test_ccsd_and_ccsd_t(get_mf, frozen):
    mf = get_mf
    ref = cc.CCSD(mf, frozen=frozen)
    ref.conv_tol = 1e-10
    ref.conv_tol_normt = 1e-8
    ref.kernel()
    et_ref = ref.ccsd_t()

    mycc = mpicc.RCCSD(mf, frozen=frozen)
    mycc.conv_tol = 1e-10
    mycc.conv_tol_normt = 1e-8
    mycc.kernel()
    et = mycc.ccsd_t()
    assert mycc.converged
    # Energies are absolute quantities, so use atol only: a relative tolerance
    # on a correlation energy of ~1e-3 Eh contributes almost nothing.  The two
    # runs follow different DIIS paths and stop anywhere within the CCSD
    # convergence thresholds (observed differences: ~1e-10 in E_corr, ~5e-12 in
    # E(T)), so the tolerance is set an order of magnitude above that.
    assert_allclose(mycc.e_corr, ref.e_corr, rtol=0, atol=1e-9)
    assert_allclose(et, et_ref, rtol=0, atol=1e-9)
