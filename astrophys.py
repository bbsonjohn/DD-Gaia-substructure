#===================================================================
#
#     Project: Direct Detection and Astrophysics
#
#     Desciption : Astrophysic phase space
#
#     Author : John Leung
#
#===================================================================

from __future__ import division    #for division operator
import math, numpy as np
from astropy import constants as const
from astropy import units as u
from astropy import coordinates as coord
from scipy.special import erf, spherical_jn
from scipy.integrate import quad, odeint, simps, nquad
import scipy.stats as stats
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline, griddata
import matplotlib.pylab as plt

from time import sleep
from tqdm import tqdm

invVelToUnitless = 3.0E5
fileDirectory = "DM_Velocity_Distribution-master/"

vMP= 235    #in km/s; from Lisanti's TASI notes
vEarth= 240 #in km/s; average yearly value; see 0808.3607, 1307.5323
vEsc= 544   #in km/s
SHMpp_eta_default = 0.3
nlb_eta_default = 0.42 #0.66 
#----------------------------------------------------------------------------------------
class EarthVelocity(object):
    #in (cartesian) galactic coordinates

    def __init__(self):
        
        self._month_transit = np.linspace(0.,365.,13)
        self._days_of_year = np.linspace(0.,365.,366)
        
        ep_rot_1 = np.array([0.9931,0.1170,-0.01032]) # 1st quadrant
        ep_rot_2 = np.array([-0.0670,0.4927,-0.8676]) # 2nd quadrant
        ep_rot_1_b = np.arcsin(ep_rot_1[2])
        ep_rot_1_l = np.arccos(ep_rot_1[0]/np.cos(ep_rot_1_b))
        ep_rot_2_b = np.arcsin(ep_rot_2[2])
        ep_rot_2_l = np.arccos(ep_rot_2[0]/np.cos(ep_rot_2_b))

        ep_rot_1_gal = coord.Galactic(l=ep_rot_1_l*u.rad, b=ep_rot_1_b*u.rad,distance=1000.*u.kpc)
        ep_rot_2_gal = coord.Galactic(l=ep_rot_2_l*u.rad, b=ep_rot_2_b*u.rad,distance=1000.*u.kpc)

        ep_rot_1_galcen = ep_rot_1_gal.transform_to(coord.Galactocentric)
        ep_rot_2_galcen = ep_rot_2_gal.transform_to(coord.Galactocentric)


        self._ep_rot_1_galcen_vec = np.array([ep_rot_1_galcen.transform_to(coord.Galactocentric).x/(1000.*u.kpc), \
                               ep_rot_1_galcen.transform_to(coord.Galactocentric).y/(1000.*u.kpc), \
                               ep_rot_1_galcen.transform_to(coord.Galactocentric).z/(1000.*u.kpc)])
        self._ep_rot_2_galcen_vec = np.array([ep_rot_2_galcen.transform_to(coord.Galactocentric).x/(1000.*u.kpc), \
                               ep_rot_2_galcen.transform_to(coord.Galactocentric).y/(1000.*u.kpc), \
                               ep_rot_2_galcen.transform_to(coord.Galactocentric).z/(1000.*u.kpc)])
        return
                               

    def f_v_earth(self, day, coord = None):
        v_LSR = np.array([0. , 235., 0.]) # local standard of rest
        v_sun = np.array([8.5, 13.38, 6.49]) # sun's peculiar velocity
        year = 365. # length of a year in days
        v_earth_o = 29.79 # earth orbital speed
        t0 = 79.55 # vernal equinox
        
        vec_indx = np.nan
        if coord == 'x' or coord == 'r':
            vec_indx = 0
        elif coord == 'y' or coord == 'the':
            vec_indx = 1
        elif coord == 'z' or coord == 'phi':
            vec_indx = 2
    
        v_e = v_LSR[vec_indx] + v_sun[vec_indx] + v_earth_o*(self._ep_rot_1_galcen_vec[vec_indx]*np.cos( (2*np.pi/year)*day-t0) \
                             + self._ep_rot_2_galcen_vec[vec_indx]*np.sin( (2*np.pi/year)*day-t0) )
        return v_e


    def f_v_earth_monthly_r(self, month='avg'):
        if month=='avg':
            return np.mean(self.f_v_earth(self._days_of_year,'x'))
        elif type(month) is int:
            month = [month]
            
        _f_v_earth_monthly_r = np.empty(len(month))
        for i, imonth in enumerate(month):
            indx_month = (self._days_of_year > self._month_transit[imonth])*(self._days_of_year < self._month_transit[imonth+1])
            _f_v_earth_monthly_r[i] = np.sum(self.f_v_earth(self._days_of_year[indx_month],'x'))/(1.*np.sum(indx_month))
        return _f_v_earth_monthly_r
                   
    def f_v_earth_monthly_the(self, month='avg'):
        if month=='avg':
            return np.mean(self.f_v_earth(self._days_of_year,'y'))
        elif type(month) is int:
            month = [month]

        _f_v_earth_monthly_the = np.empty(len(month))
        for i, imonth in enumerate(month):
            indx_month = (self._days_of_year > self._month_transit[imonth])*(self._days_of_year < self._month_transit[imonth+1])
            _f_v_earth_monthly_the[i] = np.sum(self.f_v_earth(self._days_of_year[indx_month],'y'))/(1.*np.sum(indx_month))
        return _f_v_earth_monthly_the
            
    def f_v_earth_monthly_phi(self, month='avg'):
        if month=='avg':
            return np.mean(self.f_v_earth(self._days_of_year,'z'))
        elif type(month) is int:
            month = [month]

        _f_v_earth_monthly_phi = np.empty(len(month))
        for i, imonth in enumerate(month):
            indx_month = (self._days_of_year > self._month_transit[imonth])*(self._days_of_year < self._month_transit[imonth+1])
            _f_v_earth_monthly_phi[i] = np.sum(self.f_v_earth(self._days_of_year[indx_month],'z'))/(1.*np.sum(indx_month))
        return _f_v_earth_monthly_phi

    
#----------------------------------------------------------------------------------------

def velDM_min(ER, mT, muT):
    return np.sqrt(mT*ER/(2*pow(muT, 2)))*invVelToUnitless

class VelocityFunctions(object):
    def __init__(self):
        vsteps = 100
        v_range = np.linspace(-vEsc-vEarth,vEsc+vEarth, vsteps)
        half_v_range = np.linspace(0.,vEsc+vEarth, vsteps)
        
        fileDirectory = "DM_Velocity_Distribution-master/"
        fileName = "f_v_substructure_normalized.txt"
        
        f_halo_v = self.load_fv('f_v_halo_normalized.txt')
        f_subst_v = self.load_fv('f_v_substructure_normalized.txt')
        v_file = np.linspace(0.01,800,vsteps)
        dv = v_file[1] - v_file[0]
        
        g_halo_nlb_grid = np.empty_like(v_file)
        g_subst_nlb_grid = np.empty_like(v_file)
        for i, v_i in enumerate(v_file):
            g_halo_nlb_grid[i] = np.sum((v_file > v_i)*f_halo_v(v_file)/v_file)*dv
            g_subst_nlb_grid[i] = np.sum((v_file > v_i)*f_subst_v(v_file)/v_file)*dv
        
        #self.int_velDM0_full_nlb = self.load_fv(fileName, fileDirectory)
        self.int_velDM0_halo_nlb_full = interp1d(v_file, g_halo_nlb_grid, bounds_error = False, fill_value = (g_halo_nlb_grid[0],0))
        self.int_velDM0_subst_nlb_full = interp1d(v_file, g_subst_nlb_grid, bounds_error = False, fill_value = (g_subst_nlb_grid[0],0))
        
        vx_range = np.empty([vsteps**3])
        vy_range = np.empty([vsteps**3])
        vz_range = np.empty([vsteps**3])
                
        array_count = 0
        for vi in v_range:
            for vj in v_range:
                for vk in v_range:
                    vx_range[array_count] = vi
                    vy_range[array_count] = vj
                    vz_range[array_count] = vk
                    array_count = array_count + 1
        
        int_velDM0_dft_grid = np.empty([len(half_v_range)])
        int_velDM0_shm_grid = np.empty([len(half_v_range)])
        int_velDM0_shm_Jun_grid = np.empty([len(half_v_range)])
        int_velDM0_shm_Dec_grid = np.empty([len(half_v_range)])
        int_velDM0_halo_grid = np.empty([len(half_v_range)])
        int_velDM0_ssg_grid = np.empty([len(half_v_range)])
        int_velDM0_halo_nlb_full_grid = np.empty([len(half_v_range)])
        int_velDM0_subst_nlb_full_grid = np.empty([len(half_v_range)])
        int_velDM0_halo_nlb_full_Jun_grid = np.empty([len(half_v_range)])
        int_velDM0_subst_nlb_full_Jun_grid = np.empty([len(half_v_range)])
        int_velDM0_halo_nlb_full_Dec_grid = np.empty([len(half_v_range)])
        int_velDM0_subst_nlb_full_Dec_grid = np.empty([len(half_v_range)])
        norm = (2*(vEsc+vEarth)/vsteps)**3

        array_count = 0; print "generating default halo"; sleep(0.5)
        for v_min_i in tqdm(half_v_range):
            int_velDM0_dft_grid[array_count] = norm*np.sum( f_velDM_pole_cartesian(vx_range, vy_range, vz_range, v_min_i, moment=-1, struct='default') )
            array_count = array_count + 1

        array_count = 0; print "generating SHM"; sleep(0.5)
        for v_min_i in tqdm(half_v_range):
            int_velDM0_shm_grid[array_count] = norm*np.sum( f_velDM_pole_cartesian(vx_range, vy_range, vz_range, v_min_i, moment=-1, struct='SHM') )
            array_count = array_count + 1    
         
        #array_count = 0; print "generating SHM Jun"; sleep(0.5)
        #for v_min_i in tqdm(half_v_range):
        #    int_velDM0_shm_Jun_grid[array_count] = norm*np.sum( f_velDM_pole_cartesian(vx_range, vy_range, vz_range, v_min_i, moment=-1, struct='SHM', month=6) )
        #    array_count = array_count + 1    
    
        #array_count = 0; print "generating SHM Dec"; sleep(0.5)
        #for v_min_i in tqdm(half_v_range):
        #    int_velDM0_shm_Dec_grid[array_count] = norm*np.sum( f_velDM_pole_cartesian(vx_range, vy_range, vz_range, v_min_i, moment=-1, struct='SHM', month=12) )
        #    array_count = array_count + 1    
    
        #array_count = 0; print "generating SHM++ halo"; sleep(0.5)
        #for v_min_i in tqdm(half_v_range):
        #    int_velDM0_halo_grid[array_count] = norm*np.sum( f_velDM_pole_cartesian(vx_range, vy_range, vz_range, v_min_i, moment=-1, struct='halo++') )
        #    array_count = array_count + 1


        #array_count = 0; print "generating SHM++ sausage"; sleep(0.5)
        #for v_min_i in tqdm(half_v_range):
        #    int_velDM0_ssg_grid[array_count] = norm*np.sum( f_velDM_pole_cartesian(vx_range, vy_range, vz_range, v_min_i, moment=-1, struct='sausage++') )
        #    array_count = array_count + 1
    
        #array_count = 0; print "generating NLB full substructure"; sleep(0.5)
        #for v_min_i in tqdm(half_v_range):
        #    int_velDM0_subst_nlb_full_grid[array_count] = norm*np.sum( f_velDM_pole_cartesian(vx_range, vy_range, vz_range, v_min_i, moment=-1, struct='subst_NLB') )
        #    array_count = array_count + 1
    
        #array_count = 0; print "generating NLB full halo"; sleep(0.5)
        #for v_min_i in tqdm(half_v_range):
        #    int_velDM0_halo_nlb_full_grid[array_count] = norm*np.sum( f_velDM_pole_cartesian(vx_range, vy_range, vz_range, v_min_i, moment=-1, struct='halo_NLB') )
        #    array_count = array_count + 1

        #array_count = 0; print "generating NLB full substructure June"; sleep(0.5)
        #for v_min_i in tqdm(half_v_range):
        #    int_velDM0_subst_nlb_full_Jun_grid[array_count] = norm*np.sum( f_velDM_pole_cartesian(vx_range, vy_range, vz_range, v_min_i, moment=-1, struct='subst_NLB', month=6) )
        #    array_count = array_count + 1
    
        #array_count = 0; print "generating NLB full halo June"; sleep(0.5)
        #for v_min_i in tqdm(half_v_range):
        #    int_velDM0_halo_nlb_full_Jun_grid[array_count] = norm*np.sum( f_velDM_pole_cartesian(vx_range, vy_range, vz_range, v_min_i, moment=-1, struct='halo_NLB', month=6) )
        #    array_count = array_count + 1

        #array_count = 0; print "generating NLB halo December"; sleep(0.5)
        #for v_min_i in tqdm(half_v_range):
        #    int_velDM0_halo_nlb_full_Dec_grid[array_count] = norm*np.sum( f_velDM_pole_cartesian(vx_range, vy_range, vz_range, v_min_i, moment=-1, struct='halo_NLB', month=12) )
        #    array_count = array_count + 1
    
        #array_count = 0; print "generating NLB subst December"; sleep(0.5)
        #for v_min_i in tqdm(half_v_range):
        #    int_velDM0_subst_nlb_full_Dec_grid[array_count] = norm*np.sum( f_velDM_pole_cartesian(vx_range, vy_range, vz_range, v_min_i, moment=-1, struct='subst_NLB', month=12) )
        #    array_count = array_count + 1
    

        self.int_velDM0_dft = interp1d(half_v_range, int_velDM0_dft_grid, bounds_error = False, fill_value = (int_velDM0_dft_grid[0],0))
        self.int_velDM0_shm = interp1d(half_v_range, int_velDM0_shm_grid, bounds_error = False, fill_value = (int_velDM0_shm_grid[0],0))
        #self.int_velDM0_shm_Jun = interp1d(half_v_range, int_velDM0_shm_Jun_grid, bounds_error = False, fill_value = (int_velDM0_shm_Jun_grid[0],0))
        #self.int_velDM0_shm_Dec = interp1d(half_v_range, int_velDM0_shm_Dec_grid, bounds_error = False, fill_value = (int_velDM0_shm_Dec_grid[0],0))
        #self.int_velDM0_halo = interp1d(half_v_range, int_velDM0_halo_grid, bounds_error = False, fill_value = (int_velDM0_halo_grid[0],0))
        #self.int_velDM0_ssg = interp1d(half_v_range, int_velDM0_ssg_grid, bounds_error = False, fill_value = (int_velDM0_ssg_grid[0],0))
        #self.int_velDM0_halo_nlb_full = interp1d(half_v_range, int_velDM0_halo_nlb_full_grid, bounds_error = False, fill_value = (int_velDM0_halo_nlb_full_grid[0],0))
        #self.int_velDM0_subst_nlb_full = interp1d(half_v_range, int_velDM0_subst_nlb_full_grid, bounds_error = False, fill_value = (int_velDM0_subst_nlb_full_grid[0],0))
        #self.int_velDM0_halo_nlb_full_Jun = interp1d(half_v_range, int_velDM0_halo_nlb_full_Jun_grid, bounds_error = False, fill_value = (int_velDM0_halo_nlb_full_Jun_grid[0],0))
        #self.int_velDM0_subst_nlb_full_Jun = interp1d(half_v_range, int_velDM0_subst_nlb_full_Jun_grid, bounds_error = False, fill_value = (int_velDM0_subst_nlb_full_Jun_grid[0],0))
        #self.int_velDM0_halo_nlb_full_Dec = interp1d(half_v_range, int_velDM0_halo_nlb_full_Dec_grid, bounds_error = False, fill_value = (int_velDM0_halo_nlb_full_Dec_grid[0],0))
        #self.int_velDM0_subst_nlb_full_Dec = interp1d(half_v_range, int_velDM0_subst_nlb_full_Dec_grid, bounds_error = False, fill_value = (int_velDM0_subst_nlb_full_Dec_grid[0],0))

        
        half_v_range = np.linspace(0.,vEsc+vEarth, vsteps)
        int_velDM2_dft_grid = np.empty([len(half_v_range)])
        int_velDM2_shm_grid = np.empty([len(half_v_range)])
        int_velDM2_shm_Jun_grid = np.empty([len(half_v_range)])
        int_velDM2_shm_Dec_grid = np.empty([len(half_v_range)])
        int_velDM2_halo_grid = np.empty([len(half_v_range)])
        int_velDM2_ssg_grid = np.empty([len(half_v_range)])
        int_velDM2_halo_nlb_full_grid = np.empty([len(half_v_range)])
        int_velDM2_subst_nlb_full_grid = np.empty([len(half_v_range)])
        int_velDM2_halo_nlb_full_Jun_grid = np.empty([len(half_v_range)])
        int_velDM2_subst_nlb_full_Jun_grid = np.empty([len(half_v_range)])
        int_velDM2_halo_nlb_full_Dec_grid = np.empty([len(half_v_range)])
        int_velDM2_subst_nlb_full_Dec_grid = np.empty([len(half_v_range)])
        norm = (2*(vEsc+vEarth)/vsteps)**3

        array_count = 0; print "generating default halo"; sleep(0.5)
        for v_min_i in tqdm(half_v_range):
            int_velDM2_dft_grid[array_count] = norm*np.sum( f_velDM_pole_cartesian(vx_range, vy_range, vz_range, v_min_i, moment=1, struct='default') )
            array_count = array_count + 1

        array_count = 0; print "generating SHM"; sleep(0.5)
        for v_min_i in tqdm(half_v_range):
            int_velDM2_shm_grid[array_count] = norm*np.sum( f_velDM_pole_cartesian(vx_range, vy_range, vz_range, v_min_i, moment=1, struct='SHM') )
            array_count = array_count + 1    

        #array_count = 0; print "generating SHM Jun"; sleep(0.5)
        #for v_min_i in tqdm(half_v_range):
        #    int_velDM2_shm_Jun_grid[array_count] = norm*np.sum( f_velDM_pole_cartesian(vx_range, vy_range, vz_range, v_min_i, moment=1, struct='SHM', month=6) )
        #    array_count = array_count + 1    
    
        #array_count = 0; print "generating SHM Dec"; sleep(0.5)
        #for v_min_i in tqdm(half_v_range):
        #    int_velDM2_shm_Dec_grid[array_count] = norm*np.sum( f_velDM_pole_cartesian(vx_range, vy_range, vz_range, v_min_i, moment=1, struct='SHM', month=12) )
        #    array_count = array_count + 1    
    
        #array_count = 0; print "generating SHM++ halo"; sleep(0.5)
        #for v_min_i in tqdm(half_v_range):
        #    int_velDM2_halo_grid[array_count] = norm*np.sum( f_velDM_pole_cartesian(vx_range, vy_range, vz_range, v_min_i, moment=1, struct='halo++') )
        #    array_count = array_count + 1

        #array_count = 0; print "generating SHM++ sausage"; sleep(0.5)
        #for v_min_i in tqdm(half_v_range):
        #    int_velDM2_ssg_grid[array_count] = norm*np.sum( f_velDM_pole_cartesian(vx_range, vy_range, vz_range, v_min_i, moment=1, struct='sausage++') )
        #    array_count = array_count + 1
    
        array_count = 0; print "generating NLB full substructure"; sleep(0.5)
        for v_min_i in tqdm(half_v_range):
            int_velDM2_subst_nlb_full_grid[array_count] = norm*np.sum( f_velDM_pole_cartesian(vx_range, vy_range, vz_range, v_min_i, moment=1, struct='subst_NLB') )
            array_count = array_count + 1
    
        array_count = 0; print "generating NLB full halo"; sleep(0.5)
        for v_min_i in tqdm(half_v_range):
            int_velDM2_halo_nlb_full_grid[array_count] = norm*np.sum( f_velDM_pole_cartesian(vx_range, vy_range, vz_range, v_min_i, moment=1, struct='halo_NLB') )
            array_count = array_count + 1

        #array_count = 0; print "generating NLB full substructure June"; sleep(0.5)
        #for v_min_i in tqdm(half_v_range):
        #    int_velDM2_subst_nlb_full_Jun_grid[array_count] = norm*np.sum( f_velDM_pole_cartesian(vx_range, vy_range, vz_range, v_min_i, moment=1, struct='subst_NLB', month=6) )
        #    array_count = array_count + 1
    
        #array_count = 0; print "generating NLB full halo June"; sleep(0.5)
        #for v_min_i in tqdm(half_v_range):
        #    int_velDM2_halo_nlb_full_Jun_grid[array_count] = norm*np.sum( f_velDM_pole_cartesian(vx_range, vy_range, vz_range, v_min_i, moment=1, struct='halo_NLB', month=6) )
        #    array_count = array_count + 1

        #array_count = 0; print "generating NLB halo December"; sleep(0.5)
        #for v_min_i in tqdm(half_v_range):
        #    int_velDM2_halo_nlb_full_Dec_grid[array_count] = norm*np.sum( f_velDM_pole_cartesian(vx_range, vy_range, vz_range, v_min_i, moment=1, struct='halo_NLB', month=12) )
        #    array_count = array_count + 1
    
        #array_count = 0; print "generating NLB subst December"; sleep(0.5)
        #for v_min_i in tqdm(half_v_range):
        #    int_velDM2_subst_nlb_full_Dec_grid[array_count] = norm*np.sum( f_velDM_pole_cartesian(vx_range, vy_range, vz_range, v_min_i, moment=1, struct='subst_NLB', month=12) )
        #    array_count = array_count + 1
    
        h_halo_nlb_grid = np.empty_like(v_file)
        h_subst_nlb_grid = np.empty_like(v_file)
        for i, v_i in enumerate(v_file):
            h_halo_nlb_grid[i] = np.sum((v_file > v_i)*f_halo_v(v_file)*v_file)*dv
            h_subst_nlb_grid[i] = np.sum((v_file > v_i)*f_subst_v(v_file)*v_file)*dv
            
        self.int_velDM2_halo_nlb_full = interp1d(v_file, h_halo_nlb_grid, bounds_error = False, fill_value = (h_halo_nlb_grid[0],0))
        self.int_velDM2_subst_nlb_full = interp1d(v_file, h_subst_nlb_grid, bounds_error = False, fill_value = (h_subst_nlb_grid[0],0))
        #self.int_velDM0_full_nlb = self.load_fv(fileName, fileDirectory)
        #self.int_velDM0_halo_nlb_full = interp1d(v_file, g_halo_nlb_grid, bounds_error = False, fill_value = (g_halo_nlb_grid[0],0))
        #self.int_velDM0_subst_nlb_full = interp1d(v_file, g_subst_nlb_grid, bounds_error = False, fill_value = (g_subst_nlb_grid[0],0))
        self.int_velDM2_dft = interp1d(half_v_range, int_velDM2_dft_grid, bounds_error = False, fill_value = (int_velDM2_dft_grid[0],0))
        self.int_velDM2_shm = interp1d(half_v_range, int_velDM2_shm_grid, bounds_error = False, fill_value = (int_velDM2_shm_grid[0],0))
        #self.int_velDM2_shm_Jun = interp1d(half_v_range, int_velDM2_shm_Jun_grid, bounds_error = False, fill_value = (int_velDM2_shm_Jun_grid[0],0))
        #self.int_velDM2_shm_Dec = interp1d(half_v_range, int_velDM2_shm_Dec_grid, bounds_error = False, fill_value = (int_velDM2_shm_Dec_grid[0],0))
        #self.int_velDM2_halo = interp1d(half_v_range, int_velDM2_halo_grid, bounds_error = False, fill_value = (int_velDM2_halo_grid[0],0))
        #self.int_velDM2_ssg = interp1d(half_v_range, int_velDM2_ssg_grid, bounds_error = False, fill_value = (int_velDM2_ssg_grid[0],0))
        #self.int_velDM2_halo_nlb_full = interp1d(half_v_range, int_velDM2_halo_nlb_full_grid, bounds_error = False, fill_value = (int_velDM2_halo_nlb_full_grid[0],0))
        #self.int_velDM2_subst_nlb_full = interp1d(half_v_range, int_velDM2_subst_nlb_full_grid, bounds_error = False, fill_value = (int_velDM2_subst_nlb_full_grid[0],0))
        #self.int_velDM2_halo_nlb_full_Jun = interp1d(half_v_range, int_velDM2_halo_nlb_full_Jun_grid, bounds_error = False, fill_value = (int_velDM2_halo_nlb_full_Jun_grid[0],0))
        #self.int_velDM2_subst_nlb_full_Jun = interp1d(half_v_range, int_velDM2_subst_nlb_full_Jun_grid, bounds_error = False, fill_value = (int_velDM2_subst_nlb_full_Jun_grid[0],0))
        #self.int_velDM2_halo_nlb_full_Dec = interp1d(half_v_range, int_velDM2_halo_nlb_full_Dec_grid, bounds_error = False, fill_value = (int_velDM2_halo_nlb_full_Dec_grid[0],0))
        #self.int_velDM2_subst_nlb_full_Dec = interp1d(half_v_range, int_velDM2_subst_nlb_full_Dec_grid, bounds_error = False, fill_value = (int_velDM2_subst_nlb_full_Dec_grid[0],0))


    def load_fv(self, file_name, directory_name="M_Velocity_Distribution-master/", velocity_replace=None):

        v_gvmin, f_v_gvmin_grid = np.loadtxt(fileDirectory+file_name, unpack=True)
        fv_out = interp1d(v_gvmin, f_v_gvmin_grid, bounds_error = False, fill_value = (f_v_gvmin_grid[0],0))

        if velocity_replace is not None:
            velocity_replace = fv_out
            
        return fv_out


    def fv_select(self, v_min, v_funct='default', v_moment='-1', nlb_eta=nlb_eta_default, SHMpp_eta=SHMpp_eta_default):
        
        if v_moment=='-1':
            if (v_funct == 'default'):
                #rhoDM= 0.4
                vel_integral = self.int_velDM0_dft(v_min)
            elif v_funct == 'default_analytic':
                #rhoDM= 0.4
                vel_integral = self.int_velDM0_avg(v_min)#quad(f_velDM_0, v_min, vEsc, args=(vEarth,))
            elif v_funct == 'SHM':
                #rhoDM= 0.3
                vel_integral = self.int_velDM0_shm(v_min)#quad(f_velDM_0, v_min, vEsc, args=(vEarth,))
            elif v_funct == 'SHM_Jun':
                #rhoDM= 0.3
                vel_integral = self.int_velDM0_shm_Jun(v_min)#quad(f_velDM_0, v_min, vEsc, args=(vEarth,))
            elif v_funct == 'SHM_Dec':
                #rhoDM= 0.3
                vel_integral = self.int_velDM0_shm_Dec(v_min)#quad(f_velDM_0, v_min, vEsc, args=(vEarth,))
            elif v_funct == 'halo++':
                #rhoDM= 0.55
                #eta = SHMpp_eta
                vel_integral = self.int_velDM0_halo(v_min)
            elif v_funct == 'sausage++':
                #rhoDM= 0.55
                #eta = SHMpp_eta
                vel_integral = self.int_velDM0_ssg(v_min)
            elif v_funct == 'SHM++':
                #rhoDM= 0.55
                eta = SHMpp_eta
                vel_integral = (1.- eta)*self.int_velDM0_halo(v_min) + eta*self.int_velDM0_ssg(v_min)
            elif v_funct == 'NLB_halo':
                eta = nlb_eta
                vel_integral = self.int_velDM0_halo_nlb(v_min)
            elif v_funct == 'NLB_subst':
                eta = nlb_eta
                vel_integral = self.int_velDM0_subst_nlb(v_min)
            elif v_funct == 'NLB':
                eta = nlb_eta
                vel_integral = eta*self.int_velDM0_halo_nlb(v_min) + (1.- eta)*self.int_velDM0_subst_nlb(v_min)
            elif v_funct == 'NLB_halo_full':
                eta = nlb_eta
                vel_integral = self.int_velDM0_halo_nlb_full(v_min)
            elif v_funct == 'NLB_subst_full':
                eta = nlb_eta
                vel_integral = self.int_velDM0_subst_nlb_full(v_min)
            elif v_funct == 'NLB_full':
                eta = nlb_eta
                vel_integral = eta*self.int_velDM0_halo_nlb_full(v_min) + (1.- eta)*self.int_velDM0_subst_nlb_full(v_min)
            elif v_funct == 'NLB_halo_full_Jun':
                eta = nlb_eta
                vel_integral = self.int_velDM0_halo_nlb_full_Jun(v_min)
            elif v_funct == 'NLB_subst_full_Jun':
                eta = nlb_eta
                vel_integral = self.int_velDM0_subst_nlb_full_Jun(v_min)
            elif v_funct == 'NLB_full_Jun':
                eta = nlb_eta
                vel_integral = eta*self.int_velDM0_halo_nlb_full_Jun(v_min) + (1.- eta)*self.int_velDM0_subst_nlb_full_Jun(v_min)
            elif v_funct == 'NLB_halo_full_Dec':
                eta = nlb_eta
                vel_integral = self.int_velDM0_halo_nlb_full_Dec(v_min)
            elif v_funct == 'NLB_subst_full_Dec':
                eta = nlb_eta
                vel_integral = self.int_velDM0_subst_nlb_full_Dec(v_min)
            elif v_funct == 'NLB_full_Dec':
                eta = nlb_eta
                vel_integral = eta*self.int_velDM0_halo_nlb_full_Dec(v_min) + (1.- eta)*self.int_velDM0_subst_nlb_full_Dec(v_min)
            else:
                print 'HALO MODEL NOT FOUND'
                return np.nan

        if v_moment=='1':
            if (v_funct == 'default'):
                #rhoDM= 0.4
                vel_integral = self.int_velDM2_dft(v_min)
            elif v_funct == 'default_analytic':
                #rhoDM= 0.4
                vel_integral = self.int_velDM2_avg(v_min)#quad(f_velDM_0, v_min, vEsc, args=(vEarth,))
            elif v_funct == 'SHM':
                #rhoDM= 0.3
                vel_integral = self.int_velDM2_shm(v_min)#quad(f_velDM_0, v_min, vEsc, args=(vEarth,))
            elif v_funct == 'SHM_Jun':
                #rhoDM= 0.3
                vel_integral = self.int_velDM2_shm_Jun(v_min)#quad(f_velDM_0, v_min, vEsc, args=(vEarth,))
            elif v_funct == 'SHM_Dec':
                #rhoDM= 0.3
                vel_integral = self.int_velDM2_shm_Dec(v_min)#quad(f_velDM_0, v_min, vEsc, args=(vEarth,))
            elif v_funct == 'halo++':
                #rhoDM= 0.55
                #eta = SHMpp_eta
                vel_integral = self.int_velDM2_halo(v_min)
            elif v_funct == 'sausage++':
                #rhoDM= 0.55
                #eta = SHMpp_eta
                vel_integral = self.int_velDM2_ssg(v_min)
            elif v_funct == 'SHM++':
                #rhoDM= 0.55
                eta = SHMpp_eta
                vel_integral = (1.- eta)*self.int_velDM2_halo(v_min) + eta*self.int_velDM2_ssg(v_min)
            elif v_funct == 'NLB_halo':
                eta = nlb_eta
                vel_integral = self.int_velDM2_halo_nlb(v_min)
            elif v_funct == 'NLB_subst':
                eta = nlb_eta
                vel_integral = self.int_velDM2_subst_nlb(v_min)
            elif v_funct == 'NLB':
                eta = nlb_eta
                vel_integral = eta*self.int_velDM2_halo_nlb(v_min) + (1.- eta)*self.int_velDM2_subst_nlb(v_min)
            elif v_funct == 'NLB_halo_full':
                eta = nlb_eta
                vel_integral = self.int_velDM2_halo_nlb_full(v_min)
            elif v_funct == 'NLB_subst_full':
                eta = nlb_eta
                vel_integral = self.int_velDM2_subst_nlb_full(v_min)
            elif v_funct == 'NLB_full':
                eta = nlb_eta
                vel_integral = eta*self.int_velDM2_halo_nlb_full(v_min) + (1.- eta)*self.int_velDM2_subst_nlb_full(v_min)
            elif v_funct == 'NLB_halo_full_Jun':
                eta = nlb_eta
                vel_integral = self.int_velDM2_halo_nlb_full_Jun(v_min)
            elif v_funct == 'NLB_subst_full_Jun':
                eta = nlb_eta
                vel_integral = self.int_velDM2_subst_nlb_full_Jun(v_min)
            elif v_funct == 'NLB_full_Jun':
                eta = nlb_eta
                vel_integral = eta*self.int_velDM2_halo_nlb_full_Jun(v_min) + (1.- eta)*self.int_velDM2_subst_nlb_full_Jun(v_min)
            elif v_funct == 'NLB_halo_full_Dec':
                eta = nlb_eta
                vel_integral = self.int_velDM2_halo_nlb_full_Dec(v_min)
            elif v_funct == 'NLB_subst_full_Dec':
                eta = nlb_eta
                vel_integral = self.int_velDM2_subst_nlb_full_Dec(v_min)
            elif v_funct == 'NLB_full_Dec':
                eta = nlb_eta
                vel_integral = eta*self.int_velDM2_halo_nlb_full_Dec(v_min) + (1.- eta)*self.int_velDM2_subst_nlb_full_Dec(v_min)
            else:
                print 'HALO MODEL NOT FOUND'
                return np.nan
        
        unit = pow(invVelToUnitless, -1*int(v_moment))
        return unit*vel_integral

    
    

#-------------------------------------------------------------------------------------------------------------------------------

def f_velDM_pole_cartesian(vr, vthe, vphi, vMin=0., moment=-1, struct='default', month='avg'):

    #in km/s; from Lisanti's TASI notes, see 0808.3607, 1307.5323
    
    vEarthr = 0. 
    vEarththe = 0. 
    vEarthphi = 0. 
    
    if month == 'avg':
        Earth_v = EarthVelocity()
        vEarthr = Earth_v.f_v_earth_monthly_r('avg')
        vEarththe = Earth_v.f_v_earth_monthly_the('avg')
        vEarthphi = Earth_v.f_v_earth_monthly_phi('avg')
    elif month == 'off':
        vEarthr = 0.;
        vEarththe = 0.;
        vEarthphi = 0.;
    else:
        Earth_v = EarthVelocity()
        vEarthr = Earth_v.f_v_earth_monthly_r(int(month)-1)
        vEarththe = Earth_v.f_v_earth_monthly_the(int(month)-1)
        vEarthphi = Earth_v.f_v_earth_monthly_phi(int(month)-1)
    
    
    if struct not in velocity_struct['dict']:
        print "HALO MODEL NOT FOUND"
        
    v_struct = velocity_struct[struct]
    
    v_Esc = v_struct['v_Esc']
    vMPr =  v_struct['vMPr']
    vMPthe = v_struct['vMPthe']
    vMPphi =  v_struct['vMPphi']
    vAvgr =  v_struct['vAvgr']
    vAvgthe = v_struct['vAvgPhi']
    vAvgphi =  v_struct['vAvgThe']
    N_esc = v_struct['N_esc']

    mask_Esc = (((vr+vEarthr)**2 + (vthe+vEarththe)**2 + (vphi+vEarthphi)**2) < v_Esc**2 )
    mask_VMin = (((vr)**2 + (vthe)**2 + (vphi)**2) >  vMin**2)

    N_norm = ( N_esc/np.sqrt(vMPr*vMPthe*vMPphi*(np.pi**3)) )     
    
    if struct == 'subst_NLB':
        fz = N_norm*pow(np.sqrt((vr)**2+(vthe)**2+(vphi)**2), moment)\
                *( 0.5*np.exp(-(vr+vEarthr-vAvgr)**2/vMPr) + 0.5*np.exp(-(vr+vEarthr-(-1.*vAvgr))**2/vMPr) )\
                *np.exp(-(vthe+vEarththe-vAvgthe)**2/vMPthe-(vphi+vEarthphi-vAvgphi)**2/vMPphi )
    else:
        fz = N_norm*pow(np.sqrt((vr)**2+(vthe)**2+(vphi)**2), moment)\
                *np.exp(-(vr+vEarthr)**2/vMPr-(vthe+vEarththe)**2/vMPthe-(vphi+vEarthphi)**2/vMPphi )
        
    return fz*mask_Esc*mask_VMin



def f_velDM_0(v, month='avg'):
    #for dsigma_dER \propto 1/v^2
    
    z= vEsc/vMP
    N_esc= erf(z) - 2*pow(np.pi, -1/2)*(z)*np.exp(-pow(z, 2))
    
    if v < vEsc:
        return 4*np.pi*N_esc*(v)*(1/pow(vMP*np.sqrt(np.pi), 3))  \
                        *np.exp(-(v**2 + 2*v*vEarth + vEarth**2)/pow(vMP, 2)) #+ 2*v*vEarth + vEarth**2
    else:
        return 0.0
    
def f_velDM_cartesian(vx, vy, vz, vMin, month='avg'):
    #for dsigma_dER \propto 1/v^2

    vMPx= vMP    #in km/s; from Lisanti's TASI notes
    vMPy= vMP    #in km/s; from Lisanti's TASI notes
    vMPz= vMP    #in km/s; from Lisanti's TASI notes
    vEarthx= 0 #in km/s; average yearly value; see 0808.3607, 1307.5323 and JiJi's notebook for full implementation
    vEarthy= 0 #in km/s; average yearly value; see 0808.3607, 1307.5323 and JiJi's notebook for full implementation
    vEarthz= vEarth #in km/s; average yearly value; see 0808.3607, 1307.5323 and JiJi's notebook for full implementation
    #N_esc = 1./0.983686825425
    N_esc = 1./0.957332450983

    
    mask_Esc = (((vx+vEarthx)**2 + (vy+vEarthy)**2 + (vz+vEarthz)**2) < vEsc**2 )
    mask_VMin = (((vx)**2 + (vy)**2 + (vz)**2) >  vMin**2)
    
    N_norm = (1./(vMPx*vMPy*vMPz*(np.pi)**(1.5)  ) )

    fz = N_norm*pow(np.sqrt((vx)**2+(vy)**2+(vz)**2), -1)\
                *np.exp(-(vx+vEarthx)**2/(vMPx**2)-(vy+vEarthy)**2/(vMPy**2)-(vz+vEarthz)**2/(vMPz**2) )

    return fz*mask_Esc*mask_VMin

def f_velDM_polar(v, theta,  vMin, month='avg'):
    #for dsigma_dER \propto 1/v^2

    N_esc = 1./0.984666859939
    #N_esc = 1./0.957332450983

    mask_theta = 1 - (v < vEarth)*(theta < np.pi/2.)
    mask_Esc = ( v**2 + 2*vEarth*v*np.cos(theta) + vEarth**2 < vEsc**2 )
    mask_VMin = ( v**2 >  vMin**2 )
    
    N_norm = 2.*np.pi*N_esc*(1./pow(vMP*np.sqrt(np.pi), 3))
    
    fz = N_norm*(v)*np.exp(-(v**2 + 2*vEarth*v*np.cos(theta) + vEarth**2)/pow(vMP, 2))*np.sin(theta)
    
    return fz*mask_Esc*mask_VMin
    
#---------------------------------------------------------------------------------------------------------=----

default_struct = {
        'v_Esc': vEsc,
        'vMPr': vMP**2,
        'vMPthe': vMP**2,
        'vMPphi': vMP**2,
        'vEarthr': 0., 
        'vEarththe': 0., 
        'vEarthphi': vEarth,
        'vAvgr': 0.0,
        'vAvgThe': 0.0,
        'vAvgPhi': 0.0,
        'N_esc': 1./0.957332450983  #N_esc = 1./0.983686825425
    }
SHM_struct = {
        'v_Esc': 544.,
        'vMPr': 220.**2,
        'vMPthe': 220.**2,
        'vMPphi': 220.**2,
        'vEarthr': 0., 
        'vEarththe': 0., 
        'vEarthphi': 0.,
        'vAvgr': 0.0,
        'vAvgThe': 0.0,
        'vAvgPhi': 232.0, 
        'N_esc': 1./0.969801326909
    }
halopp_struct ={ 
        'v_Esc': 528.,
        'vMPr': 233**2,
        'vMPthe': 233**2,
        'vMPphi': 233**2,
        'vAvgr': 0.0,
        'vAvgThe': 0.0,
        'vAvgPhi': 0.0,
        'N_esc': 1./0.954339013971
    }   

sausagepp_struct ={  # 1810.11468
        'v_Esc': 528,
        'v0': 233,
        'beta': 0.9,
        'vMPr': 2.*3.*(233**2)/(2.*(3.-2.*0.9)),
        'vMPphi': 2.*3.*(233**2)*(1.-0.9)/(2.*(3.-2.*0.9)),
        'vMPthe': 2.*3.*(233**2)*(1.-0.9)/(2.*(3.-2.*0.9)),
        'vAvgr': 0.0,
        'vAvgThe': 0.0,
        'vAvgPhi': 0.0,
        'N_esc': 1./0.923101066486
    }

haloNLB_struct ={
        'v_Esc': vEsc,
        'vMPr': 2.*(136.1**2),
        'vMPthe': 2.*(112.5**2),
        'vMPphi': 2.*(139.1**2),
        'vAvgr': 10.,
        'vAvgThe': 0.,
        'vAvgPhi': 24.9,
        'N_esc': 1./0.960153714808
    }
substNLB_struct ={
        'v_Esc': vEsc,
        'vMPr': 2.*(113.6**2),
        'vMPthe': 2.*(65.2**2),
        'vMPphi': 2.*(61.9**2),
        'vAvgr': 147.6,
        'vAvgThe': -2.8,
        'vAvgPhi': 27.9,
        'N_esc': 1./0.97029899998
    }
    
velocity_struct = {
        'dict': ["default", "SHM", "halo++", "sausage++", "halo_NLB", "subst_NLB"],
        'default': default_struct,
        'SHM': SHM_struct,
        'halo++': halopp_struct,
        'sausage++': sausagepp_struct,
        'halo_NLB': haloNLB_struct,
        'subst_NLB': substNLB_struct
    }