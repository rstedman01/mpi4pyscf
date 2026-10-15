#!/usr/bin/env python

'''
mpi4pyscf.pbc.df.FFTDF and AFTDF against the serial PySCF classes.

    mpiexec -np 2 pytest tests
'''

import numpy
import pytest
from numpy.testing import assert_allclose
from pyscf.pbc import gto, scf, df as pdf
from mpi4pyscf.pbc import df as mdf

ATOL = 1e-9


@pytest.fixture(scope='module')
def cell():
    return gto.M(a=numpy.eye(3)*3.5668, atom='C 0 0 0; C 0.8917 0.8917 0.8917',
                 basis='gth-szv', pseudo='gth-pade', mesh=[13]*3, verbose=0)


def _dm(cell, nk, seed=1):
    rng = numpy.random.RandomState(seed)
    nao = cell.nao_nr()
    dm = rng.rand(nk, nao, nao)
    if nk > 1:
        dm = dm + 1j * rng.rand(nk, nao, nao)
    return dm + dm.transpose(0, 2, 1).conj()


def _classes(name):
    return getattr(pdf, name), getattr(mdf, name)


@pytest.mark.parametrize('name', ['FFTDF', 'AFTDF'])
@pytest.mark.parametrize('nk', [1, 2])
def test_jk(cell, name, nk):
    ref_cls, mpi_cls = _classes(name)
    kpts = cell.make_kpts([nk, 1, 1])
    dm = _dm(cell, nk)
    ref, mydf = ref_cls(cell, kpts), mpi_cls(cell, kpts)
    for exxdiv in (None, 'ewald'):
        vj0, vk0 = ref.get_jk(dm, kpts=kpts, exxdiv=exxdiv)
        vj1, vk1 = mydf.get_jk(dm, kpts=kpts, exxdiv=exxdiv)
        assert vj1.shape == vj0.shape
        assert_allclose(vj1, vj0, rtol=0, atol=ATOL)
        # AFTDF K at k-points was once multiplied by the number of ranks
        assert_allclose(vk1, vk0, rtol=0, atol=ATOL)


@pytest.mark.parametrize('name', ['FFTDF', 'AFTDF'])
def test_jk_non_hermitian_and_band(cell, name):
    ref_cls, mpi_cls = _classes(name)
    kpts = cell.make_kpts([2, 1, 1])
    kband = cell.make_kpts([3, 1, 1])
    dm = _dm(cell, 2)
    dm_general = dm + 0.1 * _dm(cell, 2, seed=5)[:, ::-1]
    ref, mydf = ref_cls(cell, kpts), mpi_cls(cell, kpts)
    # a non-Hermitian density has a complex density on the grid
    vj0 = ref.get_jk(dm_general, hermi=0, kpts=kpts, with_k=False)[0]
    vj1 = mydf.get_jk(dm_general, hermi=0, kpts=kpts, with_k=False)[0]
    assert_allclose(vj1, vj0, rtol=0, atol=ATOL)
    vj0, vk0 = ref.get_jk(dm, kpts=kpts, kpts_band=kband, exxdiv='ewald')
    vj1, vk1 = mydf.get_jk(dm, kpts=kpts, kpts_band=kband, exxdiv='ewald')
    assert_allclose(vj1, vj0, rtol=0, atol=ATOL)
    assert_allclose(vk1, vk0, rtol=0, atol=ATOL)


@pytest.mark.parametrize('name', ['FFTDF', 'AFTDF'])
def test_single_non_gamma_kpt(cell, name):
    ref_cls, mpi_cls = _classes(name)
    kpt = cell.make_kpts([2, 1, 1])[1]
    dm = _dm(cell, 1, seed=3)[0] + 1j * _dm(cell, 1, seed=4)[0]
    dm = dm + dm.T.conj()
    ref, mydf = ref_cls(cell), mpi_cls(cell)
    vj0, vk0 = ref.get_jk(dm, kpts=kpt, exxdiv='ewald')
    vj1, vk1 = mydf.get_jk(dm, kpts=kpt, exxdiv='ewald')
    assert_allclose(vj1, vj0, rtol=0, atol=ATOL)
    assert_allclose(vk1, vk0, rtol=0, atol=ATOL)
    assert_allclose(mydf.get_pp(kpt), ref.get_pp(kpt), rtol=0, atol=ATOL)
    assert_allclose(mydf.get_nuc(kpt), ref.get_nuc(kpt), rtol=0, atol=ATOL)


@pytest.mark.parametrize('name', ['FFTDF', 'AFTDF'])
@pytest.mark.parametrize('nk', [1, 2])
def test_pp_nuc(cell, name, nk):
    ref_cls, mpi_cls = _classes(name)
    kpts = cell.make_kpts([nk, 1, 1])
    ref, mydf = ref_cls(cell, kpts), mpi_cls(cell, kpts)
    for kp in (kpts, None):
        # PySCF >= 2.10 returns (nk,nao,nao) for kpts=None
        vpp0, vpp1 = ref.get_pp(kp), mydf.get_pp(kp)
        assert vpp1.shape == vpp0.shape
        assert_allclose(vpp1, vpp0, rtol=0, atol=ATOL)
        vnuc0, vnuc1 = ref.get_nuc(kp), mydf.get_nuc(kp)
        assert vnuc1.shape == vnuc0.shape
        assert_allclose(vnuc1, vnuc0, rtol=0, atol=ATOL)


def test_krhf_energy(cell):
    # FFTDF only: the AFTDF SCF costs about a minute (serial reference included)
    # and its Fock matrix is covered by test_jk and test_pp_nuc.
    ref_cls, mpi_cls = _classes('FFTDF')
    kpts = cell.make_kpts([2, 1, 1])
    e = []
    for cls in (ref_cls, mpi_cls):
        mf = scf.KRHF(cell, kpts=kpts)
        mf.with_df = cls(cell, kpts)
        mf.conv_tol = 1e-10
        e.append(mf.kernel())
    assert_allclose(e[1], e[0], rtol=0, atol=1e-8)


def test_kpts_set_after_construction(cell):
    # kpts is a property in PySCF 2.x; the workers must see a later change
    kpts = cell.make_kpts([2, 1, 1])
    ref, mydf = pdf.FFTDF(cell), mdf.FFTDF(cell)
    ref.kpts = kpts
    mydf.kpts = kpts
    dm = _dm(cell, 2)
    assert_allclose(mydf.get_jk(dm, kpts=kpts, with_k=False)[0],
                    ref.get_jk(dm, kpts=kpts, with_k=False)[0], rtol=0, atol=ATOL)


@pytest.mark.parametrize('name', ['DF', 'GDF', 'MDF'])
def test_density_fitting_classes_are_guarded(cell, name, monkeypatch):
    # Not ported to PySCF 2.x yet: must fail loudly, not return unchecked numbers
    monkeypatch.delenv('MPI4PYSCF_EXPERIMENTAL_PBC_DF', raising=False)
    with pytest.raises(NotImplementedError):
        getattr(mdf, name)(cell)
