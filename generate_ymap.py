# Eric Baxter's code (with my filepath modifications) to generate y map from Buzzard, MICE, etc
import os
import numpy as np
from astropy.io import fits
import healpy as hp
import pdb
import pickle
from astropy import units as u
from astropy import constants as const
from scipy import interpolate as interp
from astropy.cosmology import FlatLambdaCDM
import sys
import time

import nfw_funcs as nfw

print("starting to generate y-map")

sim = sys.argv[3]
#sim = 'mice' #'mice' or 'buzzard' or 'mdpl2' or 'outerrim'
model_choice = ''
nside_ymap = 4096
save_dir = '/mnt/raid-cita/mlokken/buzzard/ymaps/'
suffix = 'nside4096_highacc'

if (sim == 'MDPL2'):
    h = 0.6777
    Om0 = 0.307115
    Ob0 = 0.048206
if (sim == 'buzzard'):
    h = 0.7
    Om0 = 0.286
    Ob0 = 0.044
if (sim == 'mice'):
    h = 0.7
    Om0 = 0.25
    Ob0 = 0.044
if (sim == 'outerrim'):
    h = 0.71
    Om0 = (0.1109 + 0.02258)/(h**2.)
    Ob0 = 0.02258/(h**2.)

nu = 150.0
cosmo = FlatLambdaCDM(H0 = h*100, Om0 = Om0, Ob0 = Ob0)
T_cmb = 2.7255*u.K

def get_halo_info_MDPL2(halo_dir, min_index, max_index):

    ra = np.array([])
    dec = np.array([])
    z = np.array([])
    mh = np.array([])
    for i in range(min_index, max_index):
        a=np.load(halo_dir + 'haloslc_%d.npy'%i)
        rai  = a[:,0] # RA of halo
        deci = a[:,1] # DEC of halo
        zi   = a[:,2] # redshift of halo
        mhi  = a[:,3] # M200c of halo in Msun/h
        ra = np.append(ra, rai)
        dec = np.append(dec, deci)
        z = np.append(z, zi)
        mh = np.append(mh, mhi)

    return ra, dec, (1./cosmo.h)*mh, z

def get_halo_info_MICE(halofile):
    #Load halo catalog
    data = fits.open(halofile)

    halo_z = data[1].data['z_cgal']
    M200c = (1./h)*(10.**(data[1].data['lmhalo']))
    ra = data[1].data['ra_gal']
    dec = data[1].data['dec_gal']

    return ra, dec, M200c, halo_z

def get_halo_info_outerrim(halofile):
    #Load halo catalog
    data = fits.open(halofile)

    halo_z = data[1].data['redshift']
    M200c = (1./h)*data[1].data['M_SO']
    ra = data[1].data['RA']
    dec = data[1].data['DEC']

    data.close()
    return ra, dec, M200c, halo_z

def get_halo_info_MICE_split(halofile, spliti, num_split, M_min):
    #Load halo catalog
    print("spliti = ", spliti, " of ", num_split)

    print("loading fits")
    data = fits.open(halofile)
    good = np.where(data[1].data['lmhalo'] > np.log10(M_min))[0]
    split_indices = np.arange(len(good)) % num_split
    in_split = np.where(split_indices == spliti-1)[0]
    M200c = (1./h)*(10.**data[1].data['lmhalo'][good[in_split]])
    halo_z = data[1].data['z_cgal'][good[in_split]]
    ra = data[1].data['ra_gal'][good[in_split]]
    dec = data[1].data['dec_gal'][good[in_split]]
    
    data.close()
    print("done with fits")

    return ra, dec, M200c, halo_z

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

    #From Joe      
    #ID < 1e8 for z<0.34 and ID > 1e8 for z>=0.34
    #ID < 1e9 for z<0.9 and ID>1e9 for z>0.9
    #good_all = np.where(((ids < 1e8) & (halo_z < 0.34)) | ((ids >= 1e8) & (ids <= 1e9) & (halo_z > 0.34) & (halo_z < 0.9))
    good_all = np.arange(len(halo_z))

    #Restrict to good      
    vec = vec[:,good_all]
    halo_z = halo_z[good_all]
    M200c = M200c[good_all]

    # rotate the positions 
    rvec = np.dot(rmat, vec).T

    # convert to angular coords
    theta, phi = hp.vec2ang(rvec)

    ra         = phi * 180. / np.pi
    dec        = 90 - theta * 180 / np.pi

    return ra, dec, M200c, halo_z, rvec

def get_all_halos_buzzard(halo_filebase, rot_file, num_list, M_min):
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


def hav(theta):
    return np.sin(theta/2.)**2.

#assumes radians
def ang_sep(ra1, dec1, ra2, dec2):
    #Haversine formula
    theta = 2.*np.arcsin(np.sqrt(hav(dec1 - dec2) + np.cos(dec1)*np.cos(dec2)*hav(ra1-ra2)))
    return theta

def eq2ang(ra,dec):
    phi = ra*np.pi/180.
    theta = (np.pi/2.) - dec*(np.pi/180.)
    return theta, phi

def ang2eq(theta,phi):
    ra = phi*180./np.pi
    dec = 90. - theta*180./np.pi
    return ra, dec

def y_to_T(y_map, nu):
    #nu in GHz
    #See carlstrom paper: https://ned.ipac.caltech.edu/level5/Sept05/Carlstrom/Carlstrom2.html
    x = const.h*nu*(1.0e9*1./u.s)/(const.k_B*T_cmb)
    f_x = x*((np.exp(x) + 1.)/(np.exp(x) - 1.)) - 4.
    T_map = T_cmb*y_map*f_x
    return T_map.value #in K

# 1109.3711 and 1608.04160
def y_profile(r_input, M_200c, R_200c, z, cosmo, model_choice):
    #r should be in physical coordinates
    #R_200c should be physical

    r_min = 0.99*np.min(r_input)
    r_max = 10.*R_200c
    num_r_table = 200#10000
    r_table = np.exp(np.linspace(np.log(r_min), np.log(r_max), num = num_r_table))

    x = r_table/R_200c
    if (model_choice == 'profilewide'):
        x *=0.75
    if (model_choice == 'profilenarrow'):
        x *= 1.25
    P_0 = 18.1*((M_200c/1.0e14)**0.154)*((1.+z)**(-0.758))
    x_c = 0.497*((M_200c/1.0e14)**(-0.00865))*((1.+z)**(0.731))
    beta = 4.35*((M_200c/1.0e14)**(0.0393))*((1.+z)**(0.415))
    alpha = 1.0
    gamma= -0.3
    Pbar_fit = P_0*((x/x_c)**gamma)*((1.+(x/x_c)**alpha)**(-beta))
    f_b = cosmo.Ob0/cosmo.Om0
    # Factor of 2 in denominator (disagrees with Vikram et al., but see Battaglia et al.)
    P_Delta = const.G*M_200c*u.Msun*200.*cosmo.critical_density(z)*f_b/(2.*R_200c*u.Mpc)
    Pbar_th = P_Delta*Pbar_fit

    if (0):
        #To compare with Battaglia paper
        fig, ax = pl.subplots(1,1)
        scaling = (1./P_Delta)*(r_table/R_200c)**3.
        ax.plot(r_table/R_200c, Pbar_th*scaling)
        ax.set_xscale('log')
        ax.set_yscale('log')
        pdb.set_trace()

    #Alternate profile
    if (model_choice == 'profile'):
        c_200c = nfw.m200_to_c200_duffy(M_200c, z, cosmo.h, 'crit')
        M_500c = nfw.convert_mass_def(M_200c, c_200c, z, 200., 500., cosmo, 'crit', 'crit')
        R_500c = nfw.M_delta_to_R_delta(M_500c*u.Msun, z, 500., cosmo, 'crit').value
        x = r_phys_table/R_500c
        h_70 = cosmo.h/0.7
        P_0 = 8.403*(h_70)**(-3./2.)
        c_500 = 1.177
        gamma = 0.3081
        alpha = 1.0510
        beta = 5.4905
        h_z = cosmo.H(z).value/100.
        alpha_p = 0.12
        alpha_p_prime = 0.1 - (alpha_p + 0.1)*((x/0.5)**3.)/(1.+(x/0.5)**3.)
        exponent = (2./3. + alpha_p + alpha_p_prime)
        norm = (h_70**2.)*(u.keV*(1./u.cm**3))*(1.65e-3)*(h_z**(8./3.))*(M_500c/(3e14*(1./h_70)))**exponent
        shape = P_0/(((c_500*x)**gamma)*(1.+ (c_500*x)**alpha)**((beta - gamma)/alpha))
        Pbar_th_arnaud = norm*shape
        P_500 = (h_70**2.)*(u.keV*(1./u.cm**3))*(1.65e-3)*(h_z**(8./3.))*(M_500c/(3e14*(1./h_70)))**(2./3.)

    #test pressure profiles
    if (0):
        P_battaglia = Pbar_th.to('keV/cm**3').value
        P_arnaud = Pbar_th_arnaud.value
        fig, ax = pl.subplots(1,1)
        ax.plot(r_table/R_500c, P_battaglia/P_500, label = 'Battaglia')
        ax.plot(r_table/R_500c, P_arnaud/P_500, label = 'Arnaud')
        #ax.plot(r_phys_table, P_battaglia, label = 'Battaglia')
        #ax.plot(r_phys_table, P_arnaud, label = 'Arnaud')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()
        pdb.set_trace()
    
    P_e = 0.518*Pbar_th
    # Electron pressure as function of distance from cluster
    P_e_interp = interp.interp1d(r_table, P_e.value, bounds_error = False, fill_value = (1.0e10,0.0))
    #physical distances
    min_chi = -4.*R_200c
    max_chi = 4.*R_200c
    num_chi = 100
    chi = np.linspace(min_chi, max_chi, num = num_chi)
    dchi = chi[1:] - chi[:-1]
    xi_table = np.zeros(num_r_table)
    #Integrate along line of sight to get Y
    for ri in range(0,num_r_table):
        integrand = P_e_interp(np.sqrt(chi**2. + r_table[ri]**2.))
        integral = np.sum(0.5*dchi*(integrand[1:] + integrand[:-1]))
        xi_table[ri] = integral
    xi_interp = interp.interp1d(r_table, xi_table)

    #NO FACTOR OF 1+Z since in physical
    normalization = const.sigma_T/(const.m_e * const.c**2.)
    try:
        xi = normalization*xi_interp(r_input)*P_e.unit*u.Mpc
    except:
        print("whooops")
        #pdb.set_trace()

    return xi.to('').value

def halos_to_Tmap_singlemz(ra, dec, halo_z, M200c, model_choice, nside_ymap):
    #coordinates
    theta, phi = eq2ang(ra, dec)
    vec = hp.ang2vec(theta, phi)

    # angular diameter distance
    z_min = 0.0
    z_max = np.max(halo_z)
    num_z = 1000
    zz = np.linspace(z_min, z_max, num = num_z)
    D_A_interp = cosmo.angular_diameter_distance(zz)
    angular_diameter_distance_interp = interp.interp1d(zz, D_A_interp)
    R200c = nfw.M_delta_to_R_delta(M200c*u.Msun, halo_z, 200., cosmo, 'crit').value
    dA = angular_diameter_distance_interp(halo_z)

    # Compute y using Battaglia profile
    # (x, M_200c, R_200c, z, cosmo)
    min_physical_dist = 0.01
    max_physical_dist = 4.*R200c
    num_physical_dist = 100
    physical_distance_table = np.exp(np.linspace(np.log(min_physical_dist), np.log(max_physical_dist), num = num_physical_dist))
    y_table = y_profile(physical_distance_table, M200c, R200c, halo_z, cosmo, model_choice)
    y_profile_interp = interp.interp1d(np.append([0.], physical_distance_table), np.append([y_table[0]], y_table))

    # Use Duffy to get concentration
    '''
    c200m = nfw.m200_to_c200_duffy(M200m, z, cosmo.h, 'mean')
    # Use concentration to convert M200m to M200c
    M200c = np.zeros(len(M200m))
    M500c = np.zeros(len(M200m))
    for ci in range(0,len(M200m)):
        M200c_i = nfw.convert_mass_def(M200m[ci], c200m[ci], z[ci], 200., 200., cosmo, 'mean', 'crit')
        M200c[ci] = M200c_i
        M500c_i = nfw.convert_mass_def(M200m[ci], c200m[ci], z[ci], 200., 500., cosmo, 'mean', 'crit')
        M500c[ci] = M500c_i

    #physical distances
    R500c = nfw.M_delta_to_R_delta(M500c*u.Msun, z, 500., cosmo, 'crit').value
    #stores Y500
    Y500_all = np.zeros(len(ra))
    '''

    # Stores total y-map
    y_sim = np.zeros(12*nside_ymap**2)

    #Stores y value at single theta
    y_test = np.zeros(len(ra))

    # Assign y-values to map
    t1 = time.time()
    for ci in range(0,len(ra)):
        '''
        if (ci % 10000 == 0):
            t2 = time.time()
            total_time = (len(ra)/10000.)*(t2-t1)
            t1 = time.time()
            print("ci = ", ci, " out of ", len(ra), " total time (hrs) = ", total_time/3600.)
        '''

        # Identify nearby pixels
        nearby_scale_factor = 3.
        nearby_angle = nearby_scale_factor*R200c/dA
        nearby_pix = hp.query_disc(nside_ymap, vec[ci,:], nearby_angle)
        nearby_theta, nearby_phi = hp.pix2ang(nside_ymap, nearby_pix)
        nearby_ra, nearby_dec = ang2eq(nearby_theta, nearby_phi)
        angular_distances = ang_sep(ra[ci]*np.pi/180., dec[ci]*np.pi/180., nearby_ra*np.pi/180., nearby_dec*np.pi/180.)
        physical_distances = dA*angular_distances
  
        #interpolate y profile onto these points
        y_ci = y_profile_interp(physical_distances)
        
        #Add contribution from cluster to ymap
        y_sim[nearby_pix] += y_ci

        if (0):
            # Get Y500
            min_r = 0.01
            max_r = R500c[ci]
            r_phys = np.linspace(min_r, max_r, num = 100)
            theta_arcmin = 60.*(180./np.pi)*r_phys/cosmo.angular_diameter_distance(z[ci]).value
            temp = y_profile(r_phys, M200c[ci], R200c[ci], halo_z[ci], cosmo, model_choice)
            #Integrate over theta_500
            to_integrate = 2.*np.pi*theta_arcmin*temp
            dtheta_arcmin = theta_arcmin[1:] - theta_arcmin[:-1]
            #arcmin^2
            Y500 = np.sum(0.5*dtheta_arcmin*(to_integrate[1:] + to_integrate[:-1]))
            Y500_all[ci] = Y500

    # Convert y to temperature
    T_map = 1.0e6*y_to_T(y_sim, nu)

    return T_map, y_sim

#test calculation
if (0):
    from astropy.cosmology import WMAP9 as cosmo
    from astropy import units 
    # y_profile(r_input, M_200c, R_200c, z, cosmo, model_choice):
    r_test  = 1.0
    M_200c_test = 1.0e15
    z_test = 0.193
    R_200c_test = ((M_200c_test*units.Msun/((4./3.)*np.pi*200.*cosmo.critical_density(z_test)))**(1./3.)).to('Mpc').value

    y_test = y_profile(r_test, M_200c_test, R_200c_test, z_test, cosmo, model_choice)
    print("y test = ", y_test)
    pdb.set_trace()

make_plots = False

if (sim == 'buzzard'):
    min_index = int(sys.argv[1])
    max_index = int(sys.argv[2])
    print("min index = ", min_index)
    print("max index = ", max_index)
    if (1):
        rot_dir = '/mnt/raid-cita/mlokken/buzzard/catalogs/halos/' # '/project2/chihway/ebaxter/Buzzard/rot_matrices/'
        rot_file = 'desy3_irot.pkl'
        halo_dir = '/mnt/raid-cita/mlokken/buzzard/catalogs/halos/Chinchilla-3/' # '/project2/chihway/ebaxter/Buzzard/Chinchilla-3/'
        halo_file = 'Chinchilla-3'
        halo_base = 'Chinchilla-3_halos.'
        #0-4,5-9,10-14
        all_num_list = np.array([0,1,2,3,4,5,6,7,17,19,21,22,23,26,27])
        num_list = all_num_list[min_index:max_index+1]
        print("loading catalog")
        ra, dec, M200c, halo_z, vec = get_all_halos_buzzard(halo_dir + halo_base, rot_dir + rot_file, num_list, 1.0e12/h)
        good = np.where(M200c > 1.0e12/h)[0]

if (sim == 'mice'):
    if (0):
        rot_file = ''
        halo_dir = '/data/ebaxter/MICE/'
        halo_file = 'mice2_halo_cat_0p25.fits'
        # load catalog
        ra, dec, M200c, halo_z = get_halo_info_MICE(halo_dir + halo_file)
        good = np.where(M200c > 1.0e12/h)[0]
    if (1):
        rot_file = ''
        halo_dir = '/project2/chihway/MICE/'
        halo_file = 'mice_all_halos.fits'
        #spliti = int(os.environ['SPLITI'])
        #num_split = int(os.environ['NUM_SPLIT'])
        spliti = int(sys.argv[1])
        num_split = int(sys.argv[2])
        M_min = 1.0e12/h
        print("Getting halo info")
        
        ra, dec, M200c, halo_z = get_halo_info_MICE_split(halo_dir + halo_file, spliti, num_split, M_min)
        good = np.where(M200c > M_min)[0]

if (sim == 'outerrim'):
    rot_file = ''
    halo_dir = '/project2/chihway/ebaxter/outer_rim/'
    halo_file = 'HaloCat_LC.fits'
    spliti = int(sys.argv[1])
    num_split = int(sys.argv[2])
    M_min = 1.0e12/h

    print("Getting halo info")    
    ra, dec, M200c, halo_z = get_halo_info_outerrim(halo_dir + halo_file)
    good = np.where(M200c > M_min)[0]
        
if (sim == 'MDPL2'):
    print("loading catalog")
    halo_dir = '/project2/chihway/sims/MDPL2/hlists/'
    min_index = int(sys.argv[1])
    max_index = int(sys.argv[2])
    print("min index = ", min_index)
    print("max index = ", max_index)
    ra, dec, M200c, halo_z = get_halo_info_MDPL2(halo_dir, min_index, max_index)
    good = np.where(M200c > 1.0e12/h)[0]
    print("Catalog loaded")
    print("N halos = ", len(good))

ra = ra[good]
dec = dec[good]
M200c = M200c[good]
halo_z = halo_z[good]

min_M200c_bin = 0.99*np.min(M200c)
max_M200c_bin = 1.01*np.max(M200c)
min_z_bin = 0.99*np.min(halo_z)
max_z_bin = 1.01*np.max(halo_z)
num_M_bins = 100
num_z_bins = 100
M200c_bins = np.exp(np.linspace(np.log(min_M200c_bin), np.log(max_M200c_bin), num = num_M_bins+1))
z_bins = np.linspace(min_z_bin, max_z_bin, num = num_z_bins+1)
print("zbins = ", z_bins)

if (0):
    z_bins = [0., 1.0]
    M200c_bins = [8.0e12, 1.0e13]
    num_M_bins = 1
    num_z_bins = 1

#output filename
if (sim == 'buzzard'):
    save_file_name = 'ymap_' + sim + '_' + str(min_index) + '_' + str(max_index) + '_NM' + str(num_M_bins) + '_Nz' + str(num_z_bins) + '_' + suffix + '_v01.fits'
if (sim == 'MDPL2'):
    save_file_name = 'ymap_' + sim + '_' + str(min_index) + '_' + str(max_index-1) + '_NM' + str(num_M_bins) + '_Nz' + str(num_z_bins) + '_' + suffix + '_v01.fits'
if (sim == 'mice'):
    save_file_name = 'ymap_' + sim + '_split' + str(spliti) + '_of_' + str(num_split) + '_NM' + str(num_M_bins) + '_Nz' + str(num_z_bins) +'_' + suffix + '_v01.fits'
if (sim == 'outerrim'):
    save_file_name = 'ymap_' + sim + '_split' + str(spliti) + '_of_' + str(num_split) + '_NM' + str(num_M_bins) + '_Nz' + str(num_z_bins) +'_' + suffix + '_v01.fits'

#Generate Tmap
y_map_total = np.zeros(12*nside_ymap**2)
halos_done = 0
for mi in range(0,num_M_bins):
    for zi in range(0,num_z_bins):
        in_bin = np.where((M200c > M200c_bins[mi]) & (M200c < M200c_bins[mi+1])  & (halo_z > z_bins[zi]) & (halo_z < z_bins[zi+1]))[0]
        if (len(in_bin) > 0):
            print("num in bin = ", len(in_bin))
        if (len(in_bin) > 0):
            mean_M200c = np.mean(M200c[in_bin])
            mean_z = np.mean(halo_z[in_bin])
            T_map, y_map = halos_to_Tmap_singlemz(ra[in_bin], dec[in_bin], mean_z, mean_M200c, model_choice, nside_ymap)
            y_map_total = y_map_total + y_map # opportunity to parallelize -- this is just a linear operation
            halos_done += len(in_bin)
            
        if (halos_done % 1000 == 0):
            print("ci = ", halos_done, " out of ", len(ra))



hp.write_map(save_dir + save_file_name, y_map_total)

