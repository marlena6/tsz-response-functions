# Modified from Zack Li

"""
ADAPTED FROM ENLIB (SIGURD NAESS)

`python pksc2hdf5.py {input.pksc} {output.h5} {num_halos}`

i.e. for a test case with only ONE HUNDRED HALOS
```
python pksc2hdf5.py /fs/lustre/project/act/mocks/websky/v0.0/halos.pksc only_a_hundred.h5 100
```

Omit the number of halos to convert *all of them*.
```
python pksc2hdf5.py /fs/lustre/project/act/mocks/websky/v0.0/halos.pksc websky_all.h5
```
"""

import sys
import numpy as np
import pyccl
import h5py
from pixell import utils, bunch, enmap
import pickle
from astropy.io import fits
import healpy as hp

def get_halo_info_buzzard(halofile, rotfile):
    #Get rotation matrix 
    rmat = np.diag([1., 1., 1.])
    if (rotfile[-4:] != 'none'):
        with open(rotfile, 'rb') as fp:
            u = pickle._Unpickler(fp)
            u.encoding = 'latin1'
            rmat = u.load()
            #rmat = pickle.load(fp)
            # h = fitsio.read(halofile) 
            # vec  = h[['PX','PY','PZ']].view((h['PX'].dtype,3)) 
    #Load halo catalog               
    data = fits.open(halofile)
    vec = np.vstack((data[1].data['PX'],data[1].data['PY'],data[1].data['PZ']))
    #ids = data[1].data['ID']  
    halo_z = data[1].data['z']
    # Convert mass to Msun from Msun/h 
    M200c = data[1].data['M200c']/h
    if (0):
        pdb.set_trace()
        R200c = data[1].data['R200c']/h
        R200c_test = nfw.M_delta_to_R_delta(M200c*u.Msun, halo_z, 200., cosmo, 'crit').value

    # rotate the positions 
    rvec = np.dot(rmat, vec).T

    # convert to angular coords
    theta, phi = hp.vec2ang(rvec)

    ra         = phi * 180. / np.pi
    dec        = 90 - theta * 180 / np.pi

    return ra, dec, M200c, halo_z, rvec


def buzzard_read(halo_filebase, rot_file, num_list, M_min):
    z = np.array([])
    M200c = np.array([])
    ra = np.array([])
    dec = np.array([])
    vec = np.array([])
    for ii in range(0,len(num_list)):
        print("loading halo file ", ii)
        halo_file = halo_filebase + str(num_list[ii]) + '.fits'
        # load catalog and rotate ra/dec  
        ra_ii, dec_ii, M200c_ii, z_ii, vec_ii = get_halo_info_buzzard(halo_file, rot_file)
        good = np.where(M200c_ii > M_min)[0]
        M200c = np.append(M200c, M200c_ii[good])
        z = np.append(z, z_ii[good])
        ra = np.append(ra, ra_ii[good])
        dec = np.append(dec, dec_ii[good])
        vec = np.append(vec, vec_ii[good])
    return ra, dec, M200c, z, vec


def websky_pkcs_read(fname, num=0, offset=0):
	"""Read rows offset:offset+num of raw data from the given pkcs file.
	if num==0, all values are read"""
	with open(fname, "r") as ifile:
		n   = np.fromfile(ifile, count=3, dtype=np.uint32)[0]-offset
		if num: n = num
		cat = np.fromfile(ifile, count=n*10, offset=offset*10*4, dtype=np.float32).reshape(n, 10)
		return cat

def websky_decode(data, cosmology, mass_interp):
	"""Go from a raw websky catalog to pos, z and m200"""
	chi     = np.sum(data.T[:3]**2,0)**0.5 # comoving Mpc
	a       = pyccl.scale_factor_of_chi(cosmology, chi)
	z       = 1/a-1
	R       = data.T[6].astype(float) * 1e6*utils.pc # m. This is *not* r200!
	rho_m   = calc_rho_c(0, cosmology)*cosmology["Omega_m"]
	m200m   = 4/3*np.pi*rho_m*R**3
	m200c    = mass_interp(m200m, z)
	ra, dec = utils.rect2ang(data.T[:3])
	return bunch.Bunch(z=z, ra=ra, dec=dec, m200m=m200m, m200c=m200c, pos=data.T[:3])


def get_H0(cosmology): return cosmology["h"]*100*1e3/(1e6*utils.pc)

def get_H(z, cosmology):
	z = np.asanyarray(z)
	return get_H0(cosmology)*pyccl.h_over_h0(cosmology, 1/(z.reshape(-1)+1)).reshape(z.shape)

def calc_rho_c(z, cosmology):
	H     = get_H(z, cosmology)
	rho_c = 3*H**2/(8*np.pi*utils.G)
	return rho_c

class MdeltaTranslator:
	def __init__(self, cosmology,
			type1="critical", delta1=200, type2="matter", delta2=200,
			zlim=[0,20], mlim=[1e11*utils.M_sun,5e16*utils.M_sun], step=0.1):
		"""Construct a functor that translates from one M_delta defintion to
		another.
		* type1, type2: Type of M_delta, e.g. m200c vs m200m.
		  * "matter": The mass inside the region where the average density is
		    delta times higher than the current matter density
		  * "critical": The same, but for the critical density instead. This
		    differs due to the presence of dark energy.
		* delta1, delta2: The delta value used in type1, type2.
		* zlim: The z-range to build the interpolator for.
		* mlim: The Mass range to build the interpolator for, in kg
		* step: The log-spacing of the interpolators.
		Some combinations of delta and type may not be supported, limited by
		support in pyccl. The main thing this object does beyond pyccl is to
		allow one to vectorize over both z and mass."""
		idef = pyccl.halos.MassDef(delta1, type1, c_m_relation="Bhattacharya13")
		odef = pyccl.halos.MassDef(delta2, type2, c_m_relation="Bhattacharya13")
		# Set up our sample grid, which will be log-spaced in both z and mass direction
		lz1, lz2 = np.log(1+np.array(zlim)) # lz = log(1+z) = -log(a)
		lm1, lm2 = np.log(np.array(mlim))   # lm = log(m)
		nz  = utils.ceil((lz2-lz1)/step)
		nm  = utils.ceil((lm2-lm1)/step)
		lzs = np.linspace(lz1, lz2, nz)
		lms = np.linspace(lm1, lm2, nm)
		olms = np.zeros((len(lzs),len(lms)))
		for ai, lz in enumerate(lzs):
			moo = np.exp(lms[-1])/utils.M_sun
			olms[ai] = idef.translate_mass(cosmology, np.exp(lms)/utils.M_sun, np.exp(-lz), odef)
		olms = np.log(olms*utils.M_sun)
		olms = utils.interpol_prefilter(olms, order=3)
		# Save parameters
		self.lz1, self.lz2, self.dlz = lz1, lz2, (lz2-lz1)/(nz-1)
		self.lm1, self.lm2, self.dlm = lm1, lm2, (lm2-lm1)/(nm-1)
		self.olms = olms
	def __call__(self, m, z):
		zpix = (np.log(1+np.array(z))-self.lz1)/self.dlz
		mpix = (np.log(m)-self.lm1)/self.dlm
		pix  = np.array([zpix,mpix])
		return np.exp(utils.interpol(self.olms, pix, order=3, prefilter=False))

rot_dir = '/mnt/raid-cita/mlokken/buzzard/catalogs/halos/' # '/project2/chihway/ebaxter/Buzzard/rot_matrices/'
rot_file = 'desy3_irot.pkl'
halo_dir = '/mnt/raid-cita/mlokken/buzzard/catalogs/halos/Chinchilla-3/' # '/project2/chihway/ebaxter/Buzzard/Chinchilla-3/'
halo_file = 'Chinchilla-3'
halo_base = 'Chinchilla-3_halos.'
all_num_list = np.array([0,1,2,3,4,5,6,7,17,19,21,22,23,26,27])
num_list = all_num_list
print("loading catalog")

outfile = '/mnt/raid-cita/mlokken/buzzard/catalogs/halos/buzzard_halos.hdf5'

dtype = np.float32
box = np.array([[-5,10],[5,-10]]) * utils.degree
shape,wcs = enmap.geometry(pos=box,res=0.5 * utils.arcmin,proj='car')
omap        = enmap.zeros(shape[-2:], wcs, dtype)
h=0.7
cosmology   = pyccl.Cosmology(Omega_c=0.24, Omega_b=0.046, h=h, sigma8=0.82, n_s=0.96, Neff=3.046, transfer_function="boltzmann_camb")
mass_interp = MdeltaTranslator(cosmology)

ra, dec, m200c, z, vec = buzzard_read(halo_dir + halo_base, rot_dir + rot_file, num_list, 1.0e12/h)
m200m    = mass_interp(m200c, z)

f = h5py.File(outfile, 'a')
f["ra"] = ra.astype(np.float32)
f["dec"] = dec.astype(np.float32)
f["m200c"] = m200c.astype(np.float32)
f["m200m"] = m200c.astype(np.float32)
f["z"] = z.astype(np.float32)
f["pos"] = vec
f.close()
