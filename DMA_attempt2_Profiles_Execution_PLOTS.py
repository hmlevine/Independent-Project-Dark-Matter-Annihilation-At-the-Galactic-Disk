### Author: Hayley Levine
### Comment: I am completely lost on what I am doing! ಥ  _______ಥ 
###          I NEED HELP!

#imports:

import sys
import time
import numpy as np
from numba import njit
from tqdm import tqdm
import matplotlib.pyplot as plt
from DMA_attempt1_PROFILES import nfw_profile, einasto_profile, moore_profile, zhao_profile, monte_carlo_uncertainty


if __name__ == "__main__":
    
    # Scale Variables
    RHO_S = 1.5e7  # Characteristic scale density (M_sun / kpc^3)
    R_S = 20.0     # Characteristic scale radius (kpc)
    SAMPLES = 500000  # Balanced resolution for rapid, accurate plotting

    # DEFINE GRID (2 parsecs down to 0.1 parsecs) I chose a custom range.
    r_grid = np.logspace(np.log10(0.002), np.log10(0.0001), 150)

    print("=" * 75)
    print("   DARK MATTER CORE SIMULATION & PLOTTING PIPELINE")
    print("=" * 75)
    print(f" Target Frame Window: 2 pc [{r_grid[0]:.4f} kpc] ──> 0.1 pc [{r_grid[-1]:.4f} kpc]")
    print(f" Resolution Scheme:  {SAMPLES:,} Monte Carlo Samples per configuration profile.")
    print("-" * 75)

    #  Running Explicit Calculations! 
    print("-> Simulating NFW Cusp Slopes...")
    nfw_m, nfw_s = monte_carlo_uncertainty(nfw_profile, r_grid, RHO_S, R_S, num_samples=SAMPLES)

    print("-> Simulating Einasto Core Curve...")
    ein_m, ein_s = monte_carlo_uncertainty(einasto_profile, r_grid, RHO_S, R_S, num_samples=SAMPLES, extra_args=(0.17,))

    print("-> Simulating Moore Profile Spikes...")
    moo_m, moo_s = monte_carlo_uncertainty(moore_profile, r_grid, RHO_S, R_S, num_samples=SAMPLES)

    print("-> Simulating Generalized Zhao Parameters...")
    zha_m, zha_s = monte_carlo_uncertainty(zhao_profile, r_grid, RHO_S, R_S, num_samples=SAMPLES, extra_args=(1.5, 1.0, 3.5))

    print("-" * 75)
    print("Computations finalized successfully. Printing localized readout summary...")
    print("-" * 75)

    # CLEAN DISCONNECTED PREVIEW DATA READOUT
    preview_indices = [0, -1]
    for idx in preview_indices:
        location_title = "Outer Core Boundary (2 pc)" if idx == 0 else "Deep Inner Focus Line (0.1 pc)"
        print(f"--- Statistics at {location_title} [r = {r_grid[idx]:.4f} kpc] ---")
        print(f"  NFW Profile:     {nfw_m[idx]:.2e} ± {nfw_s[idx]:.2e}")
        print(f"  Einasto Profile: {ein_m[idx]:.2e} ± {ein_s[idx]:.2e}")
        print(f"  Moore Profile:   {moo_m[idx]:.2e} ± {moo_s[idx]:.2e}")
        print(f"  Zhao Profile:    {zha_m[idx]:.2e} ± {zha_s[idx]:.2e}")
        print()

    print("Assembling the deep core canvas architecture...")

    # =====================================================================
    # 5. MATPLOTLIB DESIGN SETUP
    
    plt.figure(figsize=(11, 8), dpi=150)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    colors = {
        'NFW': '#1f77b4',     
        'Einasto': '#ff7f0e', 
        'Moore': '#2ca02c',   
        'Zhao': '#d62728'     
    }

    # Render Profile curves and clamp lower sigma bands to avoid log axis breakage
    plt.loglog(r_grid, nfw_m, label='NFW', color=colors['NFW'], lw=2.5)
    plt.fill_between(r_grid, np.maximum(1e-5, nfw_m - nfw_s), nfw_m + nfw_s, color=colors['NFW'], alpha=0.12)

    plt.loglog(r_grid, ein_m, label=r'Einasto ($\alpha=0.17$)', color=colors['Einasto'], lw=2.5)
    plt.fill_between(r_grid, np.maximum(1e-5, ein_m - ein_s), ein_m + ein_s, color=colors['Einasto'], alpha=0.12)

    plt.loglog(r_grid, moo_m, label='Moore', color=colors['Moore'], lw=2.5)
    plt.fill_between(r_grid, np.maximum(1e-5, moo_m - moo_s), moo_m + moo_s, color=colors['Moore'], alpha=0.12)

    plt.loglog(r_grid, zha_m, label='Zhao (1.5, 1.0, 3.5)', color=colors['Zhao'], lw=2.5)
    plt.fill_between(r_grid, np.maximum(1e-5, zha_m - zha_s), zha_m + zha_s, color=colors['Zhao'], alpha=0.12)

    # Set the limits to read moving inward (Left=2pc [0.002], Right=0.1pc [0.0001])
    plt.xlim(0.002, 0.0001)

    # Enlarge vertical scale boundaries to encompass high-density values
    plt.ylim(1e10, 1e18)

    # VERTICAL ASTRONOMICAL ANNOTATIONS LOOP
    landmarks = [
        (0.002, "Outer Core Boundary\n(2 pc)", "#7f7f7f"),         
        (0.001, "Inner Core Boundary\n(1 pc)", "#b8860b"),         
        (0.0001, "Sagittarius A* Horizon\n(0.1 pc Focus Line)", "#4b0082") 
    ]

    for r_loc, label, color in landmarks:
        plt.axvline(x=r_loc, color=color, linestyle="--", alpha=0.75, lw=1.5)
        
        # Placed at 2e15 to float clearly near the top gridline structure
        plt.text(
            r_loc * 1.03, 2e15, label, 
            color=color, fontsize=9, rotation=90, 
            weight='semibold', va='center', ha='left'
        )

    # GRID LABELS
    
    plt.title("Dark Matter Cusp Behavior: 2 pc to 0.1 pc from Core", fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("Radius from Galactic Core, $r$ [kpc]", fontsize=11)
    plt.ylabel(r"Dark Matter Density, $\rho(r)$ [$M_{\odot} / \text{kpc}^3$]", fontsize=11)

    plt.grid(True, which="both", ls="--", alpha=0.4)

    plt.tight_layout()
    
    # Save final asset image
    output_file = "dm_deep_core_focus.png"
    plt.savefig(output_file, bbox_inches='tight')
    print(f" Chart processed successfully! Graphic file asset saved to: {output_file}\n")
