### Author: Hayley Levine
### Purpose & Objective: This is the continuation of a final independent project for Spring 2025 AST 390.
###                      The goal is to establish a theoretical behavior of dark matter annihilation at the 
###                      galactic center with the use of the four equations for the dark matter halo within
###                      the galactic center radii.

## Part 1: Imports and Functions

# ᕦ(ò_óˇ)ᕤ Imports:

import numpy as np
import astropy.units as u

#import matplotlib.pyplot as plt
#from matplotlib.animation import FuncAnimation
#import matplotlib.animation as animation

from scipy.integrate import simpson

#import pandas as pd
#import time
from scipy.stats import norm
from scipy.integrate import quad
from scipy.integrate import solve_ivp
from tqdm import tqdm
from numba import njit

from astropy.coordinates import SkyCoord
from gammapy.astro.darkmatter import DarkMatterAnnihilationSpectralModel

from gammapy.astro.darkmatter import (
    DarkMatterAnnihilationSpectralModel,
    JFactory,
    PrimaryFlux,
    profiles
)


# Constants:
KPC_TO_CM = 3.086e21

#rho_s = 0.3 #Scale density (M⨀/kpc³)
#r_s = 20.0   # Scale radius in kpc
#sigma_v = 3e-26  # Annihilation cross-section (cm^3/s)
#m_chi = 100  # Dark matter particle mass (GeV)


# ಠ ▃ ಠ Functions:  # Defining dark matter density profiles

@njit()

def nfw_profile(r, rho_s, r_s):
    """Navarro-Frenk-White dark matter density profile."""
    return rho_s / ((r / r_s) * (1 + r / r_s)**2)

@njit()

def einasto_profile(r, rho_s, r_s, alpha=0.17):
    """Einasto dark matter density profile."""
    return rho_s * np.exp(- (2 / alpha) * ((r / r_s)**alpha - 1))

@njit()

def moore_profile(r, rho_s, r_s):
    """Moore dark matter density profile. """
    return rho_s / ((r / r_s)**1.5 * (1 + (r / r_s)**1.5))

@njit()

def zhao_profile(r, rho_s, r_s, gamma=1.0, alpha=1.0, beta=3.0):
    """Generalized Zhao dark matter density profile. """
#    if r == 0:
#        return 0
    return rho_s / ((r / r_s)**gamma * (1 + (r / r_s)**alpha)**((beta - gamma) / alpha))

# Numba requires explicit loops rather than keyword **kwargs...
# So..., this helps runs the actual math loop across all random samples quicker...

@njit
def _run_mc_chunk_standard(profile_func, r, rho_samples, r_s_samples, num_samples):
    """Engine for NFW and Moore (exactly 3 parameters)"""
    out = np.empty((num_samples, len(r)))
    for i in range(num_samples):
        out[i] = profile_func(r, rho_samples[i], r_s_samples[i])
    return out

@njit
def _run_mc_chunk_einasto(profile_func, r, rho_samples, r_s_samples, num_samples, alpha_val):
    """Engine for Einasto (exactly 4 parameters)"""
    out = np.empty((num_samples, len(r)))
    for i in range(num_samples):
        out[i] = profile_func(r, rho_samples[i], r_s_samples[i], alpha_val)
    return out

@njit
def _run_mc_chunk_zhao(profile_func, r, rho_samples, r_s_samples, num_samples, gamma, alpha, beta):
    """Engine for Zhao (exactly 6 parameters)"""
    out = np.empty((num_samples, len(r)))
    for i in range(num_samples):
        out[i] = profile_func(r, rho_samples[i], r_s_samples[i], gamma, alpha, beta)
    return out


#@njit()
#def _run_mc_chunk(profile_func, r, rho_samples, r_s_samples, num_samples,extra_args):
    
#    """Internal compiled loop that ealuates profiles as faster speeds!"""
    
    #creating empty matrix: shape (num_samples, len(r))
   
#    out = np.empty((num_samples, len(r)))

#    for i in range(num_samples):
        # Unpacking the extra parameters if provided:
#        if len(extra_args) == 1:       # Einasto (alpha)
#            out[i] = profile_func(r, rho_samples[i], r_s_samples[i], extra_args[0])
#        elif len(extra_args) == 3:     # Zhao (gamma, alpha, beta)
#            out[i] = profile_func(r, rho_samples[i], r_s_samples[i], extra_args[0], extra_args[1], extra_args[2])
#        else:                          # NFW and Moore
#            out[i] = profile_func(r, rho_samples[i], r_s_samples[i])
            
#    return out

# Now defining the Monte Carlo Uncertainty Calculation:

#@njit()

#def annihilation_integrand(r, profile_func, sigmav, m_chi):
#    rho = profile_func(r)
#    dr_dv = 0.5 * sigmav * (rho / m_chi)**2
#    r_cm = r * KPC_TO_CM
#    return dr_dv * (4.0 * np.pi * r_cm**2) * KPC_TO_CM


def annihilation_integrand(r, profile_func, sigmav, m_chi):
    # Call whatever profile function was passed directly
    rho = profile_func(r) 
    
    dr_dv = 0.5 * sigmav * (rho / m_chi)**2
    r_cm = r * 3.086e21
    return dr_dv * (4.0 * np.pi * r_cm**2) * 3.086e21


#def annihilation_integrand(r, profile_name, sigmav, m_chi):
#    # Select profile via standard strings
#    if profile_name == "nfw":
#        rho = nfw_profile(r)
#    elif profile_name == "einasto":
#        rho = einasto_profile(r)
#    elif profile_name == "moore":
#        rho = moore_profile(r)
#    else:
#        rho = zhao_profile(r)
        
#    dr_dv = 0.5 * sigmav * (rho / m_chi)**2
#    r_cm = r * 3.086e21
#    return dr_dv * (4.0 * np.pi * r_cm**2) * 3.086e21


















def monte_carlo_uncertainty(profile_func, r, rho_s, r_s, num_samples=100000, extra_args = None):
    """
    
    Runs the simulation with custom inputs, time benchmarks, and tqdm progress bars.
    
    Parameters:
    - extra_args: A tuple of extra shape values, e.g., (0.17,) or (1.0, 1.0, 3.0)
    
    """
    if extra_args is None:
        extra_args_arr = np.array([], dtype=np.float64)
    else:
        extra_args_arr = np.atleast_1d(np.array(extra_args, dtype=np.float64)) 

    # Updated code for the loop to handle instances where 'r' is an array of radii
    # 10% random Gaussian Variations

    rho_samples = rho_s * (1 + 0.1 * np.random.randn(num_samples)) 
    r_s_samples = r_s * (1 + 0.1 * np.random.randn(num_samples))

    chunks = 100
    chunk_size = num_samples // chunks
    samples_list = []


    # Determine the profile type by checking the function name directly in Python
    func_name = profile_func.__name__ if hasattr(profile_func, '__name__') else str(profile_func)
    

    pbar = tqdm(range(chunks), leave = False)
    for i in pbar:
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size
        
        rho_c = rho_samples[start_idx:end_idx]
        r_s_c = r_s_samples[start_idx:end_idx]
        
        # Spatial zone text transition
        r_idx = int((i / chunks) * (len(r) - 1))
        current_radius = r[-(r_idx + 1)]
        
        if current_radius > 20.0:
            zone = " Outer Halo Zone ಠ╭╮ಠ  "
        elif 5.0 <= current_radius <= 20.0:
            zone = " ☀ Solar Neighborhood"
        elif 0.1 <= current_radius < 5.0:
            zone = " Galactic Bulge (o゜▽゜)o☆  "
        else:
            zone = " Core Singular Point (ﾉ◕ヮ◕)ﾉ ︵ ┻━┻ "
            
        pbar.set_description(f"    Region: {zone} [r ~ {current_radius:.3f} kpc]")
        
        # Route to Numba chunk loops using the safe 1D array indexing
        if "einasto" in func_name:
            alpha_val = extra_args_arr[0] if len(extra_args_arr) > 0 else 0.17
            chunk_out = _run_mc_chunk_einasto(profile_func, r, rho_c, r_s_c, chunk_size, alpha_val)
        elif "zhao" in func_name:
            gamma = extra_args_arr[0] if len(extra_args_arr) > 0 else 1.0
            alpha = extra_args_arr[1] if len(extra_args_arr) > 1 else 1.0
            beta = extra_args_arr[2] if len(extra_args_arr) > 2 else 3.0
            chunk_out = _run_mc_chunk_zhao(profile_func, r, rho_c, r_s_c, chunk_size, gamma, alpha, beta)
        else:
            chunk_out = _run_mc_chunk_standard(profile_func, r, rho_c, r_s_c, chunk_size)
            
        samples_list.append(chunk_out)

#    for i in tqdm(range(chunks), desc="    Progress", leave=False):
#        start_idx = i * chunk_size
#        end_idx = start_idx + chunk_size
        
#        rho_c = rho_samples[start_idx:end_idx]
#        r_s_c = r_s_samples[start_idx:end_idx]
        
        # Pure Python handles the parameter numbers safely before Numba executes
#        if "einasto" in func_name:
#            alpha = extra_args[0] if extra_args is not None else 0.17
#            chunk_out = _run_mc_chunk_einasto(profile_func, r, rho_c, r_s_c, chunk_size, alpha)
            
#        elif "zhao" in func_name:
#            gamma, alpha, beta = extra_args if extra_args is not None else (1.0, 1.0, 3.0)
#            chunk_out = _run_mc_chunk_zhao(profile_func, r, rho_c, r_s_c, chunk_size, gamma, alpha, beta)
            
#        else: # Default fallback to NFW or Moore
#            chunk_out = _run_mc_chunk_standard(profile_func, r, rho_c, r_s_c, chunk_size)
            
#        sample_list.append(chunk_out)

        # tqdm progress bar interface initialization
#    for i in tqdm(range(chunks), desc="    Progress", leave=False):
#        start_idx = i * chunk_size
#        end_idx = start_idx + chunk_size
        
        # Calling the math engine funct for this loop iteration

#        chunk_out = _run_mc_chunk(
#            profile_func, r, 
#            rho_samples[start_idx:end_idx], 
#            r_s_samples[start_idx:end_idx], 
#            chunk_size, extra_args
#        )
#        samples_list.append(chunk_out)
 


       
    # Stack array pieces and calculate output stats
    all_samples = np.vstack(samples_list)
    mean = np.mean(all_samples, axis=0)
    std_dev = np.std(all_samples, axis=0)
    
    return mean, std_dev










