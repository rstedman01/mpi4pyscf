#!/usr/bin/env python

'''
mpi4pyscf.pbc.df.DF (GDF) against serial PySCF GDF.

The MPI implementation always uses the compensated-charge algorithm, which
PySCF selects with GDF._prefer_ccdf = True, so compare with that.

    mpiexec -np 2 pytest tests
'''

import numpy
import pytest
from numpy.testing import assert_allclose
from pyscf.pbc import gto, df as pdf
from mpi4pyscf.pbc import df as mdf


def _cell(dim=3, low_dim_ft_type=None):
    if dim == 3:
        a = numpy.eye(3) * 3.5668
    elif dim == 2:
        a = numpy.diag([3.0, 3.0, 10.0])
    else:
        a = numpy.diag([3.0, 10.0, 10.0])
    atom = ('C 0 0 0; C 0.8917 0.8917 0.8917' if dim == 3
            else 'H 0 0 0; H 0.75 0.2 0.3')
    return gto.M(a=a, atom=atom, basis='gth-szv', pseudo='gth-pade',
                 dimension=dim, low_dim_ft_type=low_dim_ft_type,
                 precision=1e-10, verbose=0)


def _dm(cell, nk, seed=1):
    rng = numpy.random.RandomState(seed)
    nao = cell.nao_nr()
    dm = rng.rand(nk, nao, nao)
    if nk > 1:
        dm = dm + 1j * rng.rand(nk, nao, nao)
    return dm + dm.transpose(0, 2, 1).conj()


def _pair(cell, kpts):
    ref = pdf.GDF(cell, kpts)
    ref._prefer_ccdf = True
    mydf = mdf.DF(cell, kpts)
    mydf._prefer_ccdf = True    # also selects the nuclear-attraction builder
    return ref, mydf


@pytest.fixture(scope='module')
def cell3d():
    return _cell()


@pytest.fixture(scope='module')
def kpts3d(cell3d):
    return cell3d.make_kpts([2, 1, 1])


@pytest.fixture(scope='module')
def dfs_k(cell3d, kpts3d):
    return _pair(cell3d, kpts3d)


@pytest.fixture(scope='module')
def dfs_gamma(cell3d):
    return _pair(cell3d, None)


def test_jk_gamma(cell3d, dfs_gamma):
    ref, mydf = dfs_gamma
    dm = _dm(cell3d, 1)[0]
    for exxdiv in (None, 'ewald'):
        vj0, vk0 = ref.get_jk(dm, exxdiv=exxdiv)
        vj1, vk1 = mydf.get_jk(dm, exxdiv=exxdiv)
        assert_allclose(vj1, vj0, rtol=0, atol=1e-8)
        assert_allclose(vk1, vk0, rtol=0, atol=1e-8)


def test_jk_kpts(cell3d, kpts3d, dfs_k):
    ref, mydf = dfs_k
    dm = _dm(cell3d, 2)
    for exxdiv in (None, 'ewald'):
        vj0, vk0 = ref.get_jk(dm, kpts=kpts3d, exxdiv=exxdiv)
        vj1, vk1 = mydf.get_jk(dm, kpts=kpts3d, exxdiv=exxdiv)
        assert_allclose(vj1, vj0, rtol=0, atol=1e-8)
        assert_allclose(vk1, vk0, rtol=0, atol=1e-8)
    # non-Hermitian density
    dm_general = dm + 0.1 * _dm(cell3d, 2, seed=5)[:, ::-1]
    vj0 = ref.get_jk(dm_general, hermi=0, kpts=kpts3d, with_k=False)[0]
    vj1 = mydf.get_jk(dm_general, hermi=0, kpts=kpts3d, with_k=False)[0]
    assert_allclose(vj1, vj0, rtol=0, atol=1e-8)


def test_single_non_gamma_kpt(cell3d, kpts3d, dfs_k):
    ref, mydf = dfs_k
    kpt = kpts3d[1]
    dm = _dm(cell3d, 1, seed=3)[0] + 1j * _dm(cell3d, 1, seed=4)[0]
    dm = dm + dm.T.conj()
    vj0, vk0 = ref.get_jk(dm, kpts=kpt, exxdiv='ewald')
    vj1, vk1 = mydf.get_jk(dm, kpts=kpt, exxdiv='ewald')
    assert_allclose(vj1, vj0, rtol=0, atol=1e-8)
    assert_allclose(vk1, vk0, rtol=0, atol=1e-8)


def test_kpts_band(cell3d, kpts3d):
    kband = cell3d.make_kpts([3, 1, 1])
    dm = _dm(cell3d, 2)
    ref, mydf = _pair(cell3d, kpts3d)    # kpts_band is fixed when building
    vj0, vk0 = ref.get_jk(dm, kpts=kpts3d, kpts_band=kband, exxdiv='ewald')
    vj1, vk1 = mydf.get_jk(dm, kpts=kpts3d, kpts_band=kband, exxdiv='ewald')
    assert_allclose(vj1, vj0, rtol=0, atol=1e-8)
    assert_allclose(vk1, vk0, rtol=0, atol=1e-8)


@pytest.mark.parametrize('nk', [1, 2])
def test_pp_nuc(cell3d, kpts3d, dfs_k, dfs_gamma, nk):
    ref, mydf = dfs_k if nk == 2 else dfs_gamma
    kp = kpts3d if nk == 2 else None
    vpp0, vpp1 = ref.get_pp(kp), mydf.get_pp(kp)
    assert vpp1.shape == vpp0.shape
    assert_allclose(vpp1, vpp0, rtol=0, atol=1e-8)
    assert_allclose(mydf.get_nuc(kp), ref.get_nuc(kp), rtol=0, atol=1e-8)


def test_eri_and_ao2mo(cell3d, kpts3d, dfs_k):
    ref, mydf = dfs_k
    nao = cell3d.nao_nr()
    kp4 = [kpts3d[0], kpts3d[0], kpts3d[1], kpts3d[1]]
    assert_allclose(mydf.get_eri(kp4), ref.get_eri(kp4), rtol=0, atol=1e-8)
    rng = numpy.random.RandomState(0)
    mo = rng.rand(nao, nao) + 1j * rng.rand(nao, nao)
    assert_allclose(mydf.ao2mo(mo, kp4), ref.ao2mo(mo, kp4), rtol=0, atol=1e-8)


def test_loop_and_naoaux(dfs_gamma):
    # loop() in serial mode must stream every process's slice of the
    # 3-index tensor to the master, in the order of the auxiliary index
    ref, mydf = dfs_gamma
    L0 = numpy.vstack([numpy.asarray(x) for x in ref.loop()])
    L1 = numpy.vstack([numpy.asarray(x) for x in mydf.loop()])
    assert L1.shape == L0.shape
    assert_allclose(L1, L0, rtol=0, atol=1e-7)
    assert mydf.get_naoaux() == ref.get_naoaux()


@pytest.mark.parametrize('dim, low_dim_ft_type, atol', [
    (1, 'inf_vacuum', 1e-9),
    (2, 'inf_vacuum', 1e-8),
    # With the truncated Coulomb kernel PySCF's own compensated-charge and
    # range-separated builders already differ by ~1e-8.
    (2, None, 1e-7),
])
def test_low_dimensional(dim, low_dim_ft_type, atol):
    cell = _cell(dim, low_dim_ft_type)
    ref = pdf.GDF(cell)
    ref._prefer_ccdf = True
    mydf = mdf.DF(cell)
    dm = _dm(cell, 1)[0]
    vj0, vk0 = ref.get_jk(dm, exxdiv=None)
    vj1, vk1 = mydf.get_jk(dm, exxdiv=None)
    assert_allclose(vj1, vj0, rtol=0, atol=atol)
    assert_allclose(vk1, vk0, rtol=0, atol=atol)


@pytest.mark.parametrize('kmesh', [None, [2, 1, 1]])
def test_scf_energy(kmesh):
    # He: one occupied s orbital, so the occupied/virtual boundary is never
    # degenerate.  (With a degenerate boundary, e.g. the first iteration of two
    # carbon atoms in a cubic cell, correct implementations legitimately follow
    # different symmetry-broken SCF paths.)
    from pyscf.pbc import scf
    cell = gto.M(a=numpy.eye(3)*3.2, atom='He 0 0 0', basis='gth-dzv',
                 pseudo='gth-pade', precision=1e-9, verbose=0)
    kpts = cell.make_kpts(kmesh) if kmesh else None
    e = []
    for cls in (pdf.GDF, mdf.DF):
        mf = scf.KRHF(cell, kpts=kpts) if kmesh else scf.RHF(cell)
        mf.with_df = cls(cell, kpts) if kmesh else cls(cell)
        mf.with_df._prefer_ccdf = True
        mf.conv_tol = 1e-10
        e.append(mf.kernel())
    assert_allclose(e[1], e[0], rtol=0, atol=1e-8)
