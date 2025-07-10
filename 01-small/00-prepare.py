
import numpy as np
import urllib.request
from pyscf import gto, scf
from pyblock2._pyscf.ao2mo import integrals as itg
from pyblock2.driver.core import DMRGDriver, SymmetryTypes

driver = DMRGDriver(scratch="/tmp", symm_type=SymmetryTypes.SU2, n_threads=64)

mol = gto.M(atom='C 0 0 0; C 0 0 1.2425', basis='6-31g', symmetry='d2h')
mf = scf.RHF(mol).run()

ncas, n_elec, spin, ecore, h1e, g2e, orb_sym = itg.get_rhf_integrals(mf, 0, None, g2e_symm=1)
idx = driver.orbital_reordering(h1e, g2e)
h1e = h1e[idx][:, idx]
g2e = g2e[idx][:, idx][:, :, idx][:, :, :, idx]

np.savez('00-ints.npz', ncas=ncas, n_elec=n_elec, spin=spin, h1e=h1e, g2e=g2e, ecore=ecore)
