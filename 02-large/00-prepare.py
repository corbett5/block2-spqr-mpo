
import numpy as np
import urllib.request
from pyblock2._pyscf.ao2mo import integrals as itg
from pyblock2.driver.core import DMRGDriver, SymmetryTypes

driver = DMRGDriver(scratch="/tmp", symm_type=SymmetryTypes.SU2, n_threads=64)

url = "https://github.com/hczhai/fe-dimer-data/raw/master/fcidump-ccpvdz/hc/FCIDUMP-21"
urllib.request.urlretrieve(url, "/tmp/FCIDUMP-21")

driver.read_fcidump(filename="/tmp/FCIDUMP-21", pg='d2h')

# orbital reordering
h1e = driver.h1e
g2e = driver.unpack_g2e(driver.g2e)
idx = driver.orbital_reordering(h1e, g2e)
h1e = h1e[idx][:, idx]
g2e = g2e[idx][:, idx][:, :, idx][:, :, :, idx]

np.savez('00-ints.npz', ncas=driver.n_sites, n_elec=driver.n_elec, spin=driver.spin, h1e=h1e, g2e=g2e, ecore=driver.ecore)
