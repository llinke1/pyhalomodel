import numpy as np
import matplotlib.pyplot as plt
#import pyhalomodel as halo
from pyhalomodel import model, window_function, profile, concentration
import pyccl as ccl

def BiHMCode_bispectrum(ks, Ms, cosmology, z=0, ingredients={'hmf': 'Sheth & Tormen (1999)', 'concentration':'Duffy et al. (2008)',
                                                               'halo definition':'Mvir', 'profile':'NFW'},
                        freePars={'eta':0, 'B':4, 'alpha1':1, 'alpha2':1, 'kstar': 0},
                        verbose=False, fastCalc=False, onlyEquilateral=True, dewiggle=True):
    hmod = model(z, cosmology["Omega_m"], name=ingredients['hmf'])
    if verbose:
        print(hmod)

    Rs = hmod.Lagrangian_radius(Ms)
    #sigmaRs = camb_results.get_sigmaR(Rs, hubble_units=True, return_R_z=False)[[z].index(z)]
    a=1/(1+z)
    sigmaRs=ccl.sigmaR(cosmology, Rs/cosmology['h'], a)
    rvs = hmod.virial_radius(Ms)

    nus=hmod._peak_height(Ms, sigmaRs)
    bloating=np.power(nus, freePars['eta'])
    cs = freePars['B']/4*concentration(Ms, z, method=ingredients['concentration'], halo_definition=ingredients['halo definition'])/bloating
    Uk = window_function(ks, rvs, cs, profile=ingredients['profile'])

    matter_profile = profile.Fourier(ks, Ms, Uk, amplitude=Ms, normalisation=hmod.rhom, mass_tracer=True) 

    if verbose:
        print(matter_profile)

    Pk_lin=ccl.linear_matter_power(cosmology, ks*cosmology['h'], a)
    
    if dewiggle:
        from scipy.ndimage import gaussian_filter1d
        dlnk = np.log(ks[1]/ks[0])    
        sigma = 0.25/dlnk

        cosmoNoWiggle=ccl.Cosmology(Omega_c=cosmology["Omega_c"], Omega_b=cosmology["Omega_b"], h=cosmology["h"], n_s=cosmology["n_s"], sigma8=cosmology["sigma8"],
                                transfer_function="eisenstein_hu_nowiggles")
        Pk_NoWiggle=ccl.linear_matter_power(cosmoNoWiggle, ks*cosmology['h'], a)

        Pk_ratio=Pk_lin/Pk_NoWiggle
        Pk_ratio=gaussian_filter1d(Pk_ratio, sigma)
        Pk_smooth=Pk_ratio*Pk_NoWiggle
        Pk_Wiggle=Pk_lin-Pk_smooth

        sigmaV=np.trapz(Pk_lin, ks)/6/np.pi**2
        f=np.exp(-ks*ks*sigmaV)

        Pk_lin=Pk_lin-(1-f)*Pk_Wiggle

    Pk_lin*=cosmology['h']**3

    Bi_3h, Bi_2h, Bi_1h, Sum=hmod.bispectrum(ks, Pk_lin, Ms, sigmaRs, {'m': matter_profile}, verbose=verbose, fastCalc=fastCalc, 
                                             onlyEquilateral=onlyEquilateral, kstar=freePars['kstar'], 
                                             f=freePars['f'], kd=freePars['kd'], nd=freePars['nd'])
    #print(freePars['alpha1'])
    Bispec=np.power(Bi_1h['m-m-m'], freePars['alpha1'])+np.power(Bi_2h['m-m-m'], freePars['alpha1'])
    Bispec=np.power(Bispec, freePars['alpha2']/freePars['alpha1'])+np.power(Bi_3h['m-m-m'], freePars['alpha2'])
    Bispec=np.power(Bispec, 1/freePars['alpha2'])
    #Bispec=Bi_1h['m-m-m']+Bi_2h['m-m-m']+Bi_3h['m-m-m']
    N=len(ks)

    Bispec=Bispec.reshape((N,N,N))
    Bi_1h=Bi_1h['m-m-m'].reshape((N,N,N))
    Bi_2h=Bi_2h['m-m-m'].reshape((N,N,N))
    Bi_3h=Bi_3h['m-m-m'].reshape((N,N,N))

    return Bispec, Bi_1h, Bi_2h, Bi_3h



    