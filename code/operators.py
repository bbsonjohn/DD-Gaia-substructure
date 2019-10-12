#===================================================================
#
#     Project: Direct Detection and Astrophysics
#
#     Desciption : A module to carry out EFT operator calculation (Fitzpatrick, et al., 1203.3542)
#
#     Author : John Leung
#
#===================================================================

from __future__ import division    #for division operator
import math, numpy as np
import astrophys as vf
from scipy.special import erf, spherical_jn
from scipy.integrate import quad, odeint, simps, nquad
import scipy.stats as stats
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline, griddata
import matplotlib.pylab as plt

from time import sleep
from tqdm import tqdm

fmtoInverseGeV= 1.0/0.197
AMUtoGeV= 0.93146
mProton= 0.938   #in GeV
mNeutron= 0.9396 #in GeV
GFermi = 1.166E-5     #in GeV^(-2)
keVtoGeV= 1e-6 
invVelToUnitless = 3.0E5

nu_mass = {
            '19F': 19.0,
            '72Ge': 72.64,
            '131Xe': 131.3
    }    

#---------------------------------------------------------------------------------------------------------
class Operator(object):
    """Signal ."""
    def __init__(self,cp=1.,cn=1.):
        self._v_funct = vf.VelocityFunctions()
        self.A = nu_mass
        self._cp = cp
        self._cn = cn
        return
            
    def F_funct(self, a, b, er, v_min, m_N=AMUtoGeV, mu_T=None, med='h', nu="131Xe", v_funct="default"):
        fv_select = self._v_funct.fv_select
        cp = self._cp
        cn = self._cn
        A = self.A[nu]
        
        er_GeV = er*keVtoGeV
        
        x = q_func(er_GeV, A*AMUtoGeV)
        factor_out = np.nan
        
        if a==1 and b==1:
            factor_out = fv_select(v_min, v_funct, '-1')*((cp**2)*F_1_1(x, 'p', 'p', nu) + \
                            + 2.*(cp*cn)*F_1_1(x, 'p', 'n', nu) + (cn**2)*F_1_1(x, 'n', 'n', nu) )
            #factor_out = fv_select(v_min, v_funct, '-1')*(m_N**2)*helm_form_factor(q_func(er*keVtoGeV, m_N*AMUtoGeV), m_N)
        elif a==3 and b==3:
            factor_out = fv_select(v_min, v_funct, '-1')*( ( (cp**2)*F_3_3_Ppp_q(x, 'p', 'p', nu) + 2*(cp*cn)*F_3_3_Ppp_q(x, 'p', 'n', nu) \
                                 + (cn**2)*F_3_3_Ppp_q(x, 'n', 'n', nu))/((m_N)**2) + ( (cp**2)*F_3_3_Sp_q(x, 'p', 'p', nu) \
                                 + 2*(cp*cn)*F_3_3_Sp_q(x, 'p', 'n', nu) + (cn**2)*F_3_3_Sp_q(x, 'n', 'n', nu))/(mu_T**2) ) \
                                 + fv_select(v_min, v_funct, '1')*( (cp**2)*F_3_3_Sp_v(x, 'p', 'p', nu) \
                                 + 2*(cp*cn)*F_3_3_Sp_v(x, 'p', 'n', nu) + (cn**2)*F_3_3_Sp_v(x, 'n', 'n', nu))
        elif a==4 and b==4:
            factor_out = fv_select(v_min, v_funct, '-1')*((cp**2)*F_4_4(x, 'p', 'p', nu) + \
                            + 2.*(cp*cn)*F_4_4(x, 'p', 'n', nu) + (cn**2)*F_4_4(x, 'n', 'n', nu) )
        elif a==5 and b==5:
            factor_out = fv_select(v_min, v_funct, '-1')*( ( (cp**2)*F_5_5_D(x, 'p', 'p', nu) + 2*(cp*cn)*F_5_5_D(x, 'p', 'n', nu) \
                                 + (cn**2)*F_5_5_D(x, 'n', 'n', nu))/((m_N)**2) + ( (cp**2)*F_5_5_M_q(x, 'p', 'p', nu) \
                                 + 2*(cp*cn)*F_5_5_M_q(x, 'p', 'n', nu) + (cn**2)*F_5_5_M_q(x, 'n', 'n', nu))/(mu_T**2) ) \
                                 + fv_select(v_min, v_funct, '1')*( (cp**2)*F_5_5_M_v(x, 'p', 'p', nu) \
                                 + 2*(cp*cn)*F_5_5_M_v(x, 'p', 'n', nu) + (cn**2)*F_5_5_M_v(x, 'n', 'n', nu))
        elif a==6 and b==6:
            factor_out = fv_select(v_min, v_funct, '-1')*((cp**2)*F_6_6(x, 'p', 'p', nu) + \
                            + 2.*(cp*cn)*F_6_6(x, 'p', 'n', nu) + (cn**2)*F_6_6(x, 'n', 'n', nu) )
        elif a==7 and b==7:
            factor_out = fv_select(v_min, v_funct, '-1')*( (cp**2)*F_7_7_q(x, 'p', 'p', nu) + 2*(cp*cn)*F_7_7_q(x, 'p', 'n', nu) \
                                 + (cn**2)*F_7_7_q(x, 'n', 'n', nu))/(mu_T**2) \
                                 + fv_select(v_min, v_funct, '1')*( (cp**2)*F_7_7_v(x, 'p', 'p', nu) \
                                 + 2*(cp*cn)*F_7_7_v(x, 'p', 'n', nu) + (cn**2)*F_7_7_v(x, 'n', 'n', nu))
        elif a==8 and b==8:
            factor_out = fv_select(v_min, v_funct, '-1')*( ( (cp**2)*F_8_8_D(x, 'p', 'p', nu) + 2*(cp*cn)*F_8_8_D(x, 'p', 'n', nu) \
                                 + (cn**2)*F_8_8_D(x, 'n', 'n', nu))/(m_N**2) + ( (cp**2)*F_8_8_M_q(x, 'p', 'p', nu) \
                                 + 2*(cp*cn)*F_8_8_M_q(x, 'p', 'n', nu) + (cn**2)*F_8_8_M_q(x, 'n', 'n', nu))/((mu_T)**2) ) \
                                 + fv_select(v_min, v_funct, '1')*( (cp**2)*F_8_8_M_v(x, 'p', 'p', nu) \
                                 + 2*(cp*cn)*F_8_8_M_v(x, 'p', 'n', nu) + (cn**2)*F_8_8_M_v(x, 'n', 'n', nu))
        elif a==9 and b==9:
            factor_out = fv_select(v_min, v_funct, '-1')*((cp**2)*F_9_9(x, 'p', 'p', nu) + \
                            + 2.*(cp*cn)*F_9_9(x, 'p', 'n', nu) + (cn**2)*F_9_9(x, 'n', 'n', nu) )
        elif a==10 and b==10:
            factor_out = fv_select(v_min, v_funct, '-1')*((cp**2)*F_10_10(x, 'p', 'p', nu) + \
                            + 2.*(cp*cn)*F_10_10(x, 'p', 'n', nu) + (cn**2)*F_10_10(x, 'n', 'n', nu) )
        elif a==11 and b==11:
            factor_out = fv_select(v_min, v_funct, '-1')*((cp**2)*F_11_11(x, 'p', 'p', nu) + \
                            + 2.*(cp*cn)*F_11_11(x, 'p', 'n', nu) + (cn**2)*F_11_11(x, 'n', 'n', nu) )
        elif a==1 and b==3:
            factor_out = fv_select(v_min, v_funct, '-1')*((cp**2)*F_1_3(x, 'p', 'p', nu) + \
                            + (cp*cn)*F_1_3(x, 'p', 'n', nu) + (cp*cn)*F_1_3(x, 'n', 'p', nu) + (cn**2)*F_1_3(x, 'n', 'n', nu) )/m_N
        elif a==4 and b==5:
            factor_out = fv_select(v_min, v_funct, '-1')*((cp**2)*F_4_5(x, 'p', 'p', nu) + \
                            + (cp*cn)*F_4_5(x, 'p', 'n', nu) + (cp*cn)*F_4_5(x, 'n', 'p', nu) + (cn**2)*F_4_5(x, 'n', 'n', nu) )/m_N
        elif a==4 and b==6:
            factor_out =  fv_select(v_min, v_funct, '-1')*((cp**2)*F_4_6(x, 'p', 'p', nu) + \
                            + (cp*cn)*F_4_6(x, 'p', 'n', nu) + (cp*cn)*F_4_6(x, 'n', 'p', nu) + (cn**2)*F_4_6(x, 'n', 'n', nu) )
        elif a==8 and b==9:
            factor_out = fv_select(v_min, v_funct, '-1')*((cp**2)*F_8_9(x, 'p', 'p', nu) + \
                            + (cp*cn)*F_8_9(x, 'p', 'n', nu) + (cp*cn)*F_8_9(x, 'n', 'p', nu) + (cn**2)*F_8_9(x, 'n', 'n', nu) )/m_N

        propagator = 1.;
        if med == 'hh' or med == 'h':
            propagator = 1.
        elif med ==  'hl' or med == 'lh':
            propagator = pow(2*A*m_N*er_GeV, -1)
        elif med == 'll' or med == 'l':
            propagator = pow(2*A*m_N*er_GeV, -2)
            
        return propagator*factor_out

    def Model_funct(self, er, v_min, m_DM, m_N=AMUtoGeV, mu_T=None, med=None, nu=None, v_funct=None):
        fv_select = self._v_funct.fv_select
        factor_out = np.nan
        A = self.A[nu]
        
        gp = 5.59; gn = -3.83; e_sq = 4.*np.pi/137.
        cp = 1.; cn = 1.
        
        er_GeV = er*keVtoGeV
        
        x = q_func(er_GeV, A*AMUtoGeV)
        qq = 2*A*m_N*er_GeV
        
        prop = 1.
        if med[-1] == 'l':
            prop = pow(qq, -1)

        if med=='mDM_h' or med=='mDM_l':
            factor_out = e_sq*(prop**2)*(  (qq**2/(2.*m_DM)**2)*fv_select(v_min, v_funct, '-1')*F_1_1(x, 'p', 'p', nu) \
                    + 4.*(fv_select(v_min, v_funct, '-1')*F_5_5_D(x, 'p', 'p', nu)/(m_N**2) \
                    + fv_select(v_min, v_funct, '-1')*F_5_5_M_q(x, 'p', 'p', nu)/(mu_T**2) \
                    + fv_select(v_min, v_funct, '1')*F_5_5_M_v(x, 'p', 'p', nu) ) \
                    - fv_select(v_min, v_funct, '-1')*qq*2.*((2*gp/m_N)*F_4_5(x, 'p', 'p', nu) + (2*gn/m_N)*F_4_5(x, 'p', 'n', nu) )
                    + fv_select(v_min, v_funct, '-1')*((1./m_DM)**2)*( (qq**2)*(gp**2)*F_4_4(x, 'p', 'p', nu) + 2.*(qq**2)*(gp*gn)*F_4_4(x, 'p', 'n', nu) \
                    + (qq**2)*(gn**2)*F_4_4(x, 'n', 'n', nu) + \
                    - 2.*qq*(gp**2)*F_4_6(x, 'p', 'p', nu) - 2.*qq*(gp*gn)*F_4_6(x, 'n', 'p', nu) \
                    - 2.*qq*(gp*gn)*F_4_6(x, 'p', 'n', nu) - 2.*qq*(gn**2)*F_4_6(x, 'n', 'n', nu) \
                    + (gp**2)*F_6_6(x, 'p', 'p', nu) + 2.*(gp*gn)*F_6_6(x, 'p', 'n', nu) + (gn**2)*F_6_6(x, 'n', 'n', nu))   )
        elif med=='mQ':
            prop = pow(2*A*m_N*er_GeV, -1)
            factor_out = (e_sq**2)*(prop**2)*fv_select(v_min, v_funct, '-1')*F_1_1(x, 'p', 'p', nu)
        elif med=='eDM_h' or med=='eDM_l':
            factor_out = 4.*e_sq*(prop**2)*fv_select(v_min, v_funct, '-1')*F_11_11(x, 'p', 'p', nu)
        elif med=='anapole_h' or med=='anapole_l':
            factor_out = e_sq*(prop**2)*(4.*(fv_select(v_min, v_funct, '-1')*(F_8_8_D(x, 'p', 'p', nu)/(m_N**2) + F_8_8_M_q(x, 'p', 'p', nu) /(mu_T**2) ) \
                                 + fv_select(v_min, v_funct, '1')*F_8_8_M_v(x, 'p', 'p', nu) )  \
                                 + fv_select(v_min, v_funct, '-1')*(gp*F_8_9(x, 'p', 'p', nu) + gn*F_8_9(x, 'p', 'n', nu) )/m_N  \
                                 + fv_select(v_min, v_funct, '-1')*((gp**2)*F_9_9(x, 'p', 'p', nu) + 2.*(gp*gn)*F_9_9(x, 'p', 'n', nu) + (gn**2)*F_9_9(x, 'n', 'n', nu)) )               
        elif med =='SS' or med =='VV':
            factor_out =  fv_select(v_min, v_funct, '-1')*((cp**2)*F_1_1(x, 'p', 'p', nu) + \
                            + 2.*(cp*cn)*F_1_1(x, 'p', 'n', nu) + (cn**2)*F_1_1(x, 'n', 'n', nu) )
        elif med[:4] =='SS_q':
            sign_q = 1.
            if med[-2] == 'n':
                sign_q = -1.
            prop = pow( 2*A*m_N*er_GeV , sign_q*int(med[-1])/2. )
            factor_out =  prop*fv_select(v_min, v_funct, '-1')*((cp**2)*F_1_1(x, 'p', 'p', nu) + \
                            + 2.*(cp*cn)*F_1_1(x, 'p', 'n', nu) + (cn**2)*F_1_1(x, 'n', 'n', nu) )
        elif med =='pSpS':
            factor_out =  fv_select(v_min, v_funct, '-1')*( (cp**2)*F_6_6(x, 'p', 'p', nu) + \
                            + 2.*(cp*cn)*F_6_6(x, 'p', 'n', nu) + (cn**2)*F_6_6(x, 'n', 'n', nu) )/(m_DM*m_N)**2
        elif med =='pVpV':
            factor_out = 16.*fv_select(v_min, v_funct, '-1')*((cp**2)*F_4_4(x, 'p', 'p', nu) + \
                            + 2.*(cp*cn)*F_4_4(x, 'p', 'n', nu) + (cn**2)*F_4_4(x, 'n', 'n', nu) )
        return factor_out


    def get_xsec_lim(self, med):
        model_plot_range ={  #low sigma, high sigma
                'h11': (-11., -7.0),
                'l11': (-14.5, -10.),
                'h33': (-5.5, -1.5),
                'l33': (-8., -4.),
                'h44': (-9., -4.5),
                'l44': (-11.5, -7.),
                'h55': (-6.5, -2.5),
                'l55': (-10., -5.),
                'h66': (-5.5, -1.5),
                'l66': (-8., -4.),
                'h77': (-5., -1.),
                'l77': (-8.5, -4.),
                'h88': (-8., -4.),
                'l88': (-10.5, -6.5),
                'h99': (-7.5, -2.5),
                'l99': (-10., -5.5),
                'h1010': (-7., -3.),
                'l1010': (-9.5, -5.5),
                'h1111': (-9.5, -5.5),
                'l1111': (-13., -8.),
                'h13': (-9.5, 5.5),
                'l13': (-12., -8.),
                'h45': (-7., -3.),
                'l45': (-9.5, -5.5),
                'h46': (-7., -3.),
                'l46': (-9.5, -5.5),
                'h89': (-8., -3.5),
                'l89': (-10., -6.),
                'mDM_h': (-6.5, -2.),
                'mDM_l': (-10., -4.5),
                'mQ': (-12.5, -8.5),
                'eDM_h': (-9.5, -5.),
                'eDM_l': (-11.5, -7.5),
                'anapole_h': (-8.5, -3.),
                'anapole_l': (-10.5, -6.),
                'SS': (-11., -7.),
                'pSpS': (-5., -0.5),
                'VV': (-11., -7.),
                'pVpV': (-9.5, -4.5),
                'SS_qn4': (-15., -10.5),
                'SS_qn3': (-13.5, -9.),
                'SS_qn2': (-13.,-8.5),
                'SS_qn1': (-12.,-7.5),
                'SS_q0': (-11.5,-7.),
                'SS_q1': (-11.,-6.5),
                'SS_q2': (-11.,-6.),
                'SS_q3': (-10.5,-5.5),
                'SS_q4': (-10., -5)
            }
            
        low_sigma, high_sigma = model_plot_range[med]
        
        return (low_sigma, high_sigma)

    def mass_func(self, mDM, target):
    #returns mass of target, and reduced mass of target-DM system; in GeV
        A = self.A[target]
        mT= A*AMUtoGeV
    
        return mT, mDM*mT/(mDM + mT)

#------------------------------------Nuclear function building blocks--------------------------------------

def q_func(ER, mT):
    #ER in keV, mT in GeV
    return np.sqrt(2*mT*ER)
    

def b_func(A_param):
    return np.sqrt(41.167/(45*pow(A_param, -1/3) - 25*pow(A_param, -2/3)))*fmtoInverseGeV 

def y_func(q, b_param):
    return pow((q*b_param)/2., 2)  

def c_chi_func(j_chi):
    return 4*j_chi*(j_chi + 1)/3
    
def helm_form_factor(q, A):

    s= 1.0*fmtoInverseGeV
    r= 1.2*pow(A, 1/3)*fmtoInverseGeV
    r0= np.sqrt(pow(r, 2) - 5*pow(s, 2))
    
    return pow((3*spherical_jn(1, q*r0))/(q*r0), 2)*np.exp(-pow(q*s, 2))

#----------------------------------------------------------------------------------------------------------
#---------------------------------------------Nuclear responses--------------------------------------------

def F_helm(y, n, nprime, A, Z):
    q = 2.*pow(y, 0.5)/b_func(A)
    if (n == 'p') and (nprime == 'p') :
        return (Z*helm_form_factor(q, A) )**2
    elif (n == 'n') and (nprime == 'n') :
        return ( (A-Z)*helm_form_factor(q, A) )**2
    else:
        return (A-Z)*Z*( helm_form_factor(q, A) )**2
    


def F_M(y, n, nprime, nuclide):
    
    if (nuclide=='19F'):
        if (n == 'p') and (nprime == 'p') :
            return np.exp(-2*y)*(81 - 96*y + 36*pow(y, 2) - 4.7*pow(y, 3) + 0.19*pow(y, 4))
        elif (n == 'n') and (nprime == 'n') :
            return np.exp(-2*y)*(100 - 130*y + 61*pow(y, 2)  - 11*pow(y, 3) + 0.73*pow(y, 4))
        else:
            return np.exp(-2*y)*(90 - 110*y + 48*pow(y, 2)  - 7.5*pow(y, 3) + 0.37*pow(y, 4))
        
    if (nuclide=='72Ge'):
        if (n == 'p') and (nprime == 'p') :
            return np.exp(-2*y)*(1000 - 2800*y + 3000*pow(y, 2) - 1500*pow(y, 3) + 400*pow(y, 4) - \
                           51*pow(y, 5) + 2.6*pow(y, 6) - 0.0069*pow(y, 7))
        elif (n == 'n') and (nprime == 'n') :
            return np.exp(-2*y)*(1600 - 4800*y + 5600*pow(y, 2)  - 3100*pow(y, 3) + 910*pow(y, 4) - \
                           130*pow(y, 5) + 7.8*pow(y, 6) - 0.039*pow(y, 7))
        else:
            return np.exp(-2*y)*(1300 - 3700*y + 4100*pow(y, 2)  - 2200*pow(y, 3) + 600*pow(y, 4) - \
                           82*pow(y, 5) + 4.5*pow(y, 6) - 0.017*pow(y, 7))
        
    if (nuclide=='131Xe'):
        if (n == 'p') and (nprime == 'p') :
            return np.exp(-2*y)*(2900 - 11000*y + 15000*pow(y, 2) - 10000*pow(y, 3) + 3800*pow(y, 4) - \
                           810*pow(y, 5) + 92*pow(y, 6) - 5.1*pow(y, 7) + 0.11*pow(y, 8))
        elif (n == 'n') and (nprime == 'n') :
            return np.exp(-2*y)*(5900 - 25000*y + 43000*pow(y, 2)  - 37000*pow(y, 3) + 19000*pow(y, 4) - \
                           5500*pow(y, 5) + 980*pow(y, 6) - 97*pow(y, 7) + 4.9*pow(y, 8) - 0.0096*pow(y, 9) + \
                           0.0061*pow(y, 10)) 
        else:
            return np.exp(-2*y)*(4200 - 16000*y + 25000*pow(y, 2)  - 20000*pow(y, 3) + 8600*pow(y, 4) - \
                           2200*pow(y, 5) + 310*pow(y, 6) - 24*pow(y, 7) + 0.83*pow(y, 8) - 0.0082*pow(y, 9))
                           
    if (nuclide=='40Ar'):
        Ar_Z = 18; Ar_A = 40
        return F_helm(y, n, nprime, Ar_A, Ar_Z)
            
        
        
def F_D(y, n, nprime, nuclide):
    
    if (nuclide=='19F'):
        if (n == 'p') and (nprime == 'p') :
            return np.exp(-2*y)*(0.0251 - 0.0201*y + 0.00401*pow(y, 2))
        elif (n == 'n') and (nprime == 'n') :
            return np.exp(-2*y)*(0.0181 - 0.0145*y + 0.00290*pow(y, 2))
        else:
            return np.exp(-2*y)*(-0.0213 + 0.0170*y - 0.00341*pow(y, 2))
        
    if (nuclide=='72Ge'):
        return 0.
        
    if (nuclide=='131Xe'):
        if (n == 'p') and (nprime == 'p') :
            return np.exp(-2*y)*(0.022 - 0.054*y + 0.053*pow(y, 2) - 0.026*pow(y, 3) + 0.0071*pow(y, 4) - 0.00098*pow(y, 5))
        elif (n == 'n') and (nprime == 'n') :
            return np.exp(-2*y)*(0.56 - 1.4*y + 1.8*pow(y, 2)  - 1.3*pow(y, 3) + 0.51*pow(y, 4) - \
                           0.11*pow(y, 5) + 0.0097*pow(y, 6) - 0.00015*pow(y, 7)) 
        else:
            return np.exp(-2*y)*(0.11 - 0.28*y + 0.31*pow(y, 2)  - 0.19*pow(y, 3) + 0.062*pow(y, 4) - \
                           0.010*pow(y, 5) + 0.00073*pow(y, 6))

def F_Sp(y, n, nprime, nuclide):
    
    if (nuclide=='19F'):
        if (n == 'p') and (nprime == 'p') :
            return np.exp(-2*y)*(1.81 - 4.85*y + 4.88*pow(y, 2) - 2.18*pow(y, 3) + 0.364*pow(y, 4))
        elif (n == 'n') and (nprime == 'n') :
            return np.exp(-2*y)*(0.000607 - 0.00136*y + 0.000266*pow(y, 2)  + 0.000550*pow(y, 3) + 0.0000997*pow(y, 4)) 
        else:
            return np.exp(-2*y)*(-0.0331 + 0.0815*y - 0.0511*pow(y, 2)  - 0.00142*pow(y, 3) + 0.00602*pow(y, 4))
    if (nuclide=='72Ge'):
        return 0.        
    if (nuclide=='131Xe'):
        if (n == 'p') and (nprime == 'p') :
            return np.exp(-2*y)*(0.00012 - 0.00089*y + 0.0015*pow(y, 2) + 0.0015*pow(y, 3) - 0.00069*pow(y, 4) - \
                           0.0012*pow(y, 5) + 0.00080*pow(y, 6) - 0.00016*pow(y, 7))
        elif (n == 'n') and (nprime == 'n') :
            return np.exp(-2*y)*(0.18 - 1.6*y + 5.8*pow(y, 2)  - 9.7*pow(y, 3) + 9.1*pow(y, 4) - \
                           4.9*pow(y, 5) + 1.4*pow(y, 6) - 0.21*pow(y, 7) + 0.012*pow(y, 8)) 
        else:
            return np.exp(-2*y)*(0.0045 - 0.039*y + 0.095*pow(y, 2)  - 0.038*pow(y, 3) - 0.077*pow(y, 4) + \
                           0.087*pow(y, 5) - 0.035*pow(y, 6) + 0.0059*pow(y, 7))

        
def F_Spp(y, n, nprime, nuclide):

    if (nuclide=='19F'):
        if (n == 'p') and (nprime == 'p') :
            return np.exp(-2*y)*(0.903 - 2.37*y + 2.35*pow(y, 2) - 1.05*pow(y, 3) + 0.175*pow(y, 4))
        elif (n == 'n') and (nprime == 'n') :
            return np.exp(-2*y)*(0.000303 - 0.00107*y + 0.00114*pow(y, 2)  - 0.000348*pow(y, 3) + 0.0000320*pow(y, 4)) 
        else:
            return np.exp(-2*y)*(-0.0166 + 0.0509*y - 0.0510*pow(y, 2)  + 0.0199*pow(y, 3) - 0.00237*pow(y, 4))
    if (nuclide=='72Ge'):
        return 0.        
    if (nuclide=='131Xe'):
        if (n == 'p') and (nprime == 'p') :
            return np.exp(-2*y)*(0.00013*pow(y, 2) - 0.00062*pow(y, 3) + 0.00088*pow(y, 4) - 0.00053*pow(y, 5) + 0.00015*pow(y, 6))
        elif (n == 'n') and (nprime == 'n') :
            return np.exp(-2*y)*(0.088 + 0.30*y - 0.23*pow(y, 2)  - 0.47*pow(y, 3) + 1.2*pow(y, 4) - \
                           1.1*pow(y, 5) + 0.44*pow(y, 6) - 0.086*pow(y, 7) + 0.0067*pow(y, 8)) 
        else:
            return np.exp(-2*y)*(0.0023 + 0.0032*y - 0.011*pow(y, 2)  - 0.00077*pow(y, 3) + 0.019*pow(y, 4) - \
                           0.018*pow(y, 5) + 0.0066*pow(y, 6) - 0.0011*pow(y, 7))
        

        
def F_Sp_D(y, n, nprime, nuclide):

    if (nuclide=='19F'):
        if (n == 'p') and (nprime == 'p') :
            return np.exp(-2*y)*(-0.213 + 0.315*y - 0.210*pow(y, 2) + 0.0382*pow(y, 3))
        elif (n == 'p') and (nprime == 'n') :
            return np.exp(-2*y)*(0.181 - 0.315*y + 0.178*pow(y, 2)  - 0.0325*pow(y, 3))
        elif (n == 'n') and (nprime == 'p') :
            return np.exp(-2*y)*(0.00390 - 0.00592*y + 0.000163*pow(y, 2)  + 0.000632*pow(y, 3))
        elif (n == 'n') and (nprime == 'n') :
            return np.exp(-2*y)*(-0.00331 + 0.00503*y - 0.000138*pow(y, 2)  - 0.000537*pow(y, 3))        

    if (nuclide=='72Ge'):
        return 0.
        
    if (nuclide=='131Xe'):
        if (n == 'p') and (nprime == 'p') :
            return np.exp(-2*y)*(0.0016 - 0.0081*y + 0.0047*pow(y, 2) + 0.0049*pow(y, 3) - 0.0061*pow(y, 4) + \
                           0.0023*pow(y, 5) - 0.00039*pow(y, 6))
        elif (n == 'p') and (nprime == 'n') :
            return np.exp(-2*y)*(0.0080 - 0.041*y + 0.027*pow(y, 2)  + 0.020*pow(y, 3) - 0.038*pow(y, 4) + \
                           0.020*pow(y, 5) - 0.0042*pow(y, 6) + 0.00032*pow(y, 7))
        elif (n == 'n') and (nprime == 'p') :
            return np.exp(-2*y)*(0.063 - 0.37*y + 0.72*pow(y, 2)  - 0.68*pow(y, 3) + 0.35*pow(y, 4) - \
                           0.098*pow(y, 5) + 0.014*pow(y, 6) - 0.00077*pow(y, 7))
        elif (n == 'n') and (nprime == 'n') :
            return np.exp(-2*y)*(0.31 - 1.9*y + 3.8*pow(y, 2)  - 4.1*pow(y, 3) + 2.5*pow(y, 4) - \
                           0.87*pow(y, 5) + 0.15*pow(y, 6) - 0.011*pow(y, 7))  
        
def F_Ppp(y, n, nprime, nuclide):
    
    if (nuclide=='19F'):
        if (n == 'p') and (nprime == 'p') :
            return np.exp(-2*y)*(0.0392 - 0.0314*y + 0.00627*pow(y, 2))
        elif (n == 'n') and (nprime == 'n') :
            return np.exp(-2*y)*(0.255 - 0.204*y + 0.0408*pow(y, 2))     
        else:
            return np.exp(-2*y)*(0.100 - 0.0800*y + 0.0160*pow(y, 2) )

    if (nuclide=='72Ge'):
        if (n == 'p') and (nprime == 'p') :
            return np.exp(-2*y)*(68. - 110.*y + 64.*pow(y, 2) - 16.*pow(y, 3) + 1.4*pow(y, 4) - \
                           0.010*pow(y, 5))
        elif (n == 'n') and (nprime == 'n') :
            return np.exp(-2*y)*(0.71 - 1.5*y + 1.2*pow(y, 2)  - 0.42*pow(y, 3) + 0.075*pow(y, 4) - \
                           0.0063*pow(y, 5) + 0.00020*pow(y, 6))
        else :
            return np.exp(-2*y)*(6.9 - 13.*y + 8.8*pow(y, 2)  - 2.7*pow(y, 3) + 0.36*pow(y, 4) - \
                           0.018*pow(y, 5))
        
    if (nuclide=='131Xe'):
        if (n == 'p') and (nprime == 'p') :
            return np.exp(-2*y)*(100. - 250.*y + 230.*pow(y, 2) - 100.*pow(y, 3) + 24.*pow(y, 4) - \
                           2.7*pow(y, 5) + 0.12*pow(y, 6))
        elif (n == 'n') and (nprime == 'n') :
            return np.exp(-2*y)*(200. - 640.*y + 780.*pow(y, 2) - 480.*pow(y, 3) + 160.*pow(y, 4) - \
                           31.*pow(y, 5) + 3.3*pow(y, 6) - 0.18*pow(y, 7) + 0.0038*pow(y, 8))
        else :
            return np.exp(-2*y)*(150. - 400.*y + 430.*pow(y, 2)  - 230.*pow(y, 3) + 64.*pow(y, 4) - \
                           9.9*pow(y, 5) + 0.75*pow(y, 6) - 0.022*pow(y, 7))  
    
def F_M_Ppp(y, n, nprime, nuclide):
    
    if (nuclide=='19F'):
        if (n == 'p') and (nprime == 'p') :
            return np.exp(-2*y)*(-1.78 + 1.77*y - 0.509*pow(y, 2) + 0.0347*pow(y, 3))
        elif (n == 'p') and (nprime == 'n') :
            return np.exp(-2*y)*(-4.55 + 4.51*y - 1.30*pow(y, 2)  + 0.0884*pow(y, 3))
        elif (n == 'n') and (nprime == 'p') :
            return np.exp(-2*y)*(-1.98 + 2.11*y - 0.697*pow(y, 2)  + 0.0675*pow(y, 3))
        elif (n == 'n') and (nprime == 'n') :
            return np.exp(-2*y)*(-5.05 + 5.39*y - 1.78*pow(y, 2)  + 0.172*pow(y, 3))        

    if (nuclide=='72Ge'):
        if (n == 'p') and (nprime == 'p') :
            return np.exp(-2*y)*(-260. + 580.*y - 460.*pow(y, 2) + 170.*pow(y, 3) - 30.*pow(y, 4) + \
                           2.0*pow(y, 5) - 0.0094*pow(y, 6))
        elif (n == 'p') and (nprime == 'n') :
            return np.exp(-2*y)*(-27. + 66.*y - 60.*pow(y, 2) + 26.*pow(y, 3) - 5.6*pow(y, 4) + \
                           0.58*pow(y, 5) - 0.023*pow(y, 6))
        elif (n == 'n') and (nprime == 'p') :
            return np.exp(-2*y)*(-330. + 760.*y - 650.*pow(y, 2) + 250.*pow(y, 3) - 47.*pow(y, 4) + \
                           3.4*pow(y, 5) - 0.020*pow(y, 6))
        elif (n == 'n') and (nprime == 'n') :
            return np.exp(-2*y)*(-34. + 87.*y - 83.*pow(y, 2) + 38.*pow(y, 3) - 8.7*pow(y, 4) + \
                           0.96*pow(y, 5) - 0.040*pow(y, 6) + 0.00010*pow(y, 7))  
        
    if (nuclide=='131Xe'):
        if (n == 'p') and (nprime == 'p') :
            return np.exp(-2*y)*(-550. + 1700.*y - 1900.*pow(y, 2) + 1100.*pow(y, 3) - 320.*pow(y, 4) + \
                           51.*pow(y, 5) - 4.0*pow(y, 6) + 0.11*pow(y, 7))
        elif (n == 'p') and (nprime == 'n') :
            return np.exp(-2*y)*(-770. - 2600.*y - 3400.*pow(y, 2) + 2200.*pow(y, 3) - 790.*pow(y, 4) + \
                           160.*pow(y, 5) - 17.*pow(y, 6) + 0.96*pow(y, 7) - 0.021*pow(y, 8))
        elif (n == 'n') and (nprime == 'p') :
            return np.exp(-2*y)*(-790. + 2600.*y - 3400.*pow(y, 2) + 2200.*pow(y, 3) - 770.*pow(y, 4) + \
                           150.*pow(y, 5) - 16.*pow(y, 6) + 0.77*pow(y, 7) - 0.0086*pow(y, 8))
        elif (n == 'n') and (nprime == 'n') :
            return np.exp(-2*y)*(-1100. + 4100.*y - 5900.*pow(y, 2) + 4400.*pow(y, 3) - 1800.*pow(y, 4) + \
                           440.*pow(y, 5) - 61.*pow(y, 6) + 4.5*pow(y, 7) - 0.16*pow(y, 8) + 0.0015*pow(y, 9))  
                           
#----------------------------------------------------------------------------------------------------------
#-------------------------------------Nuclear EFT operator Matrix elements---------------------------------
def F_1_1(q, n, nprime, nuclide): #F_m    

    A=nu_mass[nuclide]
        
    b= b_func(A)
    y= y_func(q, b)
    
    if (nuclide=='40Ar'):
        return helm_form_factor(q, A)
        
    return F_M(y, n, nprime, nuclide)


def F_3_3_Ppp_q(q, n, nprime, nuclide, j_chi=1./2.): # (C(j)/16)*(F_S" + F_S')
    
    c_chi = c_chi_func(j_chi)
    A=nu_mass[nuclide]
        
    b= b_func(A)
    y= y_func(q, b)
    
    return (q**4/(4.))*F_Ppp(y, n, nprime, nuclide)


def F_3_3_Sp_q(q, n, nprime, nuclide, j_chi=1./2.): # (C(j)/16)*(F_S" + F_S')

    c_chi = c_chi_func(j_chi)
    A=nu_mass[nuclide]
        
    b= b_func(A)
    y= y_func(q, b)
    
    return -(q**4/(4.))*F_Sp(y, n, nprime, nuclide)


def F_3_3_Sp_v(q, n, nprime, nuclide, j_chi=1./2.): # (C(j)/16)*(F_S" + F_S')
    
    c_chi = c_chi_func(j_chi)
    A=nu_mass[nuclide]
        
    b= b_func(A)
    y= y_func(q, b)
    
    return (q**2)*F_Sp(y, n, nprime, nuclide) 


def F_4_4(q, n, nprime, nuclide, j_chi=1./2.): # (C(j)/16)*(F_S" + F_S')

    c_chi = c_chi_func(j_chi)
    A=nu_mass[nuclide]
        
    b= b_func(A)
    y= y_func(q, b)
    
    return (c_chi/16.)*(F_Spp(y, n, nprime, nuclide) + F_Sp(y, n, nprime, nuclide))



def F_5_5_M_q(q, n, nprime, nuclide, j_chi=1./2.):
    
    c_chi = c_chi_func(j_chi)
    A=nu_mass[nuclide]
        
    b= b_func(A)
    y= y_func(q, b)
    
    return (c_chi/4.)*( -(q**4/(4.))*F_M(y, n, nprime, nuclide) )



def F_5_5_M_v(q, n, nprime, nuclide, j_chi=1./2.):

    c_chi = c_chi_func(j_chi)
    A=nu_mass[nuclide]
    
    b= b_func(A)
    y= y_func(q, b)
    
    return (c_chi/4.)*( (q**2)*F_M(y, n, nprime, nuclide) )



def F_5_5_D(q, n, nprime, nuclide, j_chi=1./2.):

    c_chi = c_chi_func(j_chi)
    A=nu_mass[nuclide]
        
    b= b_func(A)
    y= y_func(q, b)
    
    return (c_chi/4.)*( (q**4)*F_D(y, n, nprime, nuclide) )



def F_6_6(q, n, nprime, nuclide, j_chi=1./2.):

    c_chi = c_chi_func(j_chi)
    A=nu_mass[nuclide]
        
    b= b_func(A)
    y= y_func(q, b)
    
    return (c_chi)*(q**4/16.)*( F_Spp(y, n, nprime, nuclide) )


def F_7_7_v(q, n, nprime, nuclide, j_chi=1./2.):

    A=nu_mass[nuclide]
        
    b= b_func(A)
    y= y_func(q, b)
    
    return (1./8.)*( F_Sp(y, n, nprime, nuclide) )



def F_7_7_q(q, n, nprime, nuclide, j_chi=1./2.):

    A=nu_mass[nuclide]
    
    b= b_func(A)
    y= y_func(q, b)
    
    return (1./8.)*( -(q**2/4.)*F_Sp(y, n, nprime, nuclide) )



def F_8_8_M_q(q, n, nprime, nuclide, j_chi=1./2.):
 
    c_chi = c_chi_func(j_chi)
    A=nu_mass[nuclide]
        
    b= b_func(A)
    y= y_func(q, b)
    
    return (c_chi/4.)*( -(q**2/4.)*F_M(y, n, nprime, nuclide) )



def F_8_8_M_v(q, n, nprime, nuclide, j_chi=1./2.):

    c_chi = c_chi_func(j_chi)
    A=nu_mass[nuclide]
        
    b= b_func(A)
    y= y_func(q, b)
    
    return (c_chi/4.)*( F_M(y, n, nprime, nuclide) )



def F_8_8_D(q, n, nprime, nuclide, j_chi=1./2.):

    c_chi = c_chi_func(j_chi)
    A=nu_mass[nuclide]
        
    b= b_func(A)
    y= y_func(q, b)
    
    return (c_chi/4.)*( (q**2)*F_D(y, n, nprime, nuclide) )


def F_9_9(q, n, nprime, nuclide, j_chi=1./2.):

    c_chi = c_chi_func(j_chi)
    A=nu_mass[nuclide]
        
    b= b_func(A)
    y= y_func(q, b)
    
    return c_chi*( (q**2/16.)*F_Sp(y, n, nprime, nuclide) )



def F_10_10(q, n, nprime, nuclide, j_chi=1./2.):

    A=nu_mass[nuclide]
        
    b= b_func(A)
    y= y_func(q, b)
    
    return (q**2/4.)*F_Spp(y, n, nprime, nuclide)
    

def F_11_11(q, n, nprime, nuclide, j_chi=1./2.):
  
    c_chi = c_chi_func(j_chi)
    A=nu_mass[nuclide]
        
    b= b_func(A)
    y= y_func(q, b)
    
    return (c_chi)*(q**2/4.)*( F_M(y, n, nprime, nuclide) )



def F_1_3(q, n, nprime, nuclide, j_chi=1./2.): # (C(j)/16)*(F_S" + F_S')

    A=nu_mass[nuclide]
        
    b= b_func(A)
    y= y_func(q, b)
    
    return (q**2/2.)*F_M_Ppp(y, n, nprime, nuclide) 



def F_4_5(q, n, nprime, nuclide, j_chi=1./2.):

    c_chi = c_chi_func(j_chi)
    A=nu_mass[nuclide]
        
    b= b_func(A)
    y= y_func(q, b)
    
    return (c_chi)*(q**2/8.)*( F_Sp_D(y, n, nprime, nuclide) )



def F_4_6(q, n, nprime, nuclide, j_chi=1./2.):

    c_chi = c_chi_func(j_chi)
    A=nu_mass[nuclide]
        
    b= b_func(A)
    y= y_func(q, b)
    
    return (c_chi)*(q**2/16.)*( F_Spp(y, n, nprime, nuclide) )




def F_8_9(q, n, nprime, nuclide, j_chi=1./2.):

    c_chi = c_chi_func(j_chi)
    A=nu_mass[nuclide]
        
    b= b_func(A)
    y= y_func(q, b)
    
    return -(c_chi)*(q**2/8.)*( F_Sp_D(y, n, nprime, nuclide) )


#----------------------------------------------------------------------------------------------------------