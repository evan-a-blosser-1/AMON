"""
Lyapunov Exponent Analysis for Orbital Dynamics - Orbit_Analysis.py

ACADEMIC CITATIONS AND SOURCES:
===============================

Primary Algorithm Source:
- Wolf, A., Swift, J.B., Swinney, H.L., & Vastano, J.A. (1985). 
  "Determining Lyapunov exponents from a time series." 
  Physica D: Nonlinear Phenomena, 16(3), 285-317.
  DOI: 10.1016/0167-2789(85)90011-9

Practical Implementation:
- Rosenstein, M.T., Collins, J.J., & De Luca, C.J. (1993). 
  "A practical method for calculating largest Lyapunov exponents from 
  small data sets." Physica D: Nonlinear Phenomena, 65(1-2), 117-134.
  DOI: 10.1016/0167-2789(93)90009-P

Orbital Mechanics Theory:
- Scheeres, D.J. (2012). "Orbital Motion in Strongly Perturbed 
  Environments: Applications to Asteroid, Comet and Planetary 
  Satellite Orbiters." Springer-Praxis.

Spectral Analysis:
- Welch, P. (1967). "The use of fast Fourier transform for the estimation 
  of power spectra: a method based on time averaging over short, modified 
  periodograms." IEEE Transactions on Audio and Electroacoustics, 15(2), 70-73.

Mathematical Foundation:
- Oseledec, V.I. (1968). "A multiplicative ergodic theorem. Lyapunov 
  characteristic numbers for dynamical systems." Transactions of the Moscow 
  Mathematical Society, 19, 197-231.

See references_and_citations.py for complete bibliography.
"""
import numpy as np
from scipy import signal
def estimate_lyapunov_exponent(trajectory_data, dt):
    """
    Estimate the largest Lyapunov exponent from trajectory data
    
    Mathematical Method: Nearby Trajectory Separation
    ================================================
    
    CITATION: This implementation is based on the algorithm described in:
    Wolf, A., Swift, J.B., Swinney, H.L., & Vastano, J.A. (1985). 
    "Determining Lyapunov exponents from a time series." 
    Physica D: Nonlinear Phenomena, 16(3), 285-317.
    
    With practical improvements from:
    Rosenstein, M.T., Collins, J.J., & De Luca, C.J. (1993). 
    "A practical method for calculating largest Lyapunov exponents from 
    small data sets." Physica D: Nonlinear Phenomena, 65(1-2), 117-134.
    
    Theory:
    -------
    The largest Lyapunov exponent λ₁ is defined as:
    λ₁ = lim[t→∞] lim[δx₀→0] (1/t) ln(||δx(t)||/||δx₀||)
    
    This mathematical definition was originally formulated by Lyapunov (1892)
    and rigorously developed by Oseledec (1968).
    
    Implementation:
    --------------
    1. For reference point x_ref(t_r), find nearby point x_near(t_n)
    2. Track separation: d(t) = ||x_ref(t_r + t) - x_near(t_n + t)||
    3. Calculate local exponent: λ_local = d/dt[ln(d(t))] ≈ slope(ln(d) vs t)
    4. Average over multiple trajectory pairs: λ₁ = ⟨λ_local⟩
    
    Parameters:
    -----------
    trajectory_data : array
        Trajectory data (n_points, 6) - [x, y, z, vx, vy, vz]
    dt : float
        Time step in seconds
        
    Returns:
    --------
    lyap_estimate : float
        Estimated largest Lyapunov exponent (1/s)
        
    Mathematical Notes:
    ------------------
    • Positive λ₁: Chaotic (exponential divergence)
    • Zero λ₁: Neutral/marginal stability  
    • Negative λ₁: Stable (exponential convergence)
    • Lyapunov time: τ = 1/|λ₁| (predictability horizon)
    
    References:
    ----------
    See references_and_citations.py for complete bibliography.
    """
    n_points = len(trajectory_data)
    lyap_estimates = []
    
    # Mathematical Algorithm: Nearby Trajectory Method
    # ===============================================
    # CITATION: Algorithm based on Wolf et al. (1985) with improvements from
    # Rosenstein et al. (1993) for practical implementation from time series data.
    
    # Step 1: Sample reference points for computational efficiency
    # Theoretical basis: Statistical sampling improves robustness (Kantz, 1994)
    n_refs = min(50, n_points // 20)
    ref_indices = np.linspace(0, n_points - 500, n_refs, dtype=int)
    
    for ref_idx in ref_indices:
        ref_point = trajectory_data[ref_idx]  # Reference trajectory point x_ref(t_r)
        
        # Step 2: Find nearby point in 6D phase space
        # Mathematical constraint: ||x_ref - x_near|| = min{distances} > δ_min
        search_range = min(n_points // 10, 1000)
        end_idx = min(ref_idx + search_range, n_points)
        
        # Euclidean distance in 6D phase space: d = √(Σᵢ(xᵢ - x'ᵢ)²)
        distances = np.sqrt(np.sum((trajectory_data[ref_idx:end_idx] - ref_point)**2, axis=1))
        
        # Exclude self-distance (= 0) and enforce minimum separation threshold
        if len(distances) < 2:
            continue
            
        valid_distances = distances[1:]  # Exclude self
        if len(valid_distances) == 0 or np.min(valid_distances) < 1e-8:  # δ_min = 10⁻⁸
            continue
            
        nearest_local_idx = np.argmin(valid_distances) + 1
        nearest_idx = ref_idx + nearest_local_idx
        
        if nearest_idx >= n_points - 100:
            continue
        
        # Step 3: Track separation evolution d(t) = ||x_ref(t_r + t) - x_near(t_n + t)||
        # Mathematical model: d(t) ≈ d₀ * exp(λ₁ * t) for chaotic systems
        separations = []  # ln(d(t))
        times = []        # t
        max_evolution = min(100, n_points - max(ref_idx, nearest_idx))
        
        for i in range(max_evolution):
            if ref_idx + i < n_points and nearest_idx + i < n_points:
                # Calculate separation at time t = i * dt
                sep = np.linalg.norm(trajectory_data[ref_idx + i] - trajectory_data[nearest_idx + i])
                if sep > 1e-10:  # Avoid ln(0)
                    separations.append(np.log(sep))  # ln(d(t))
                    times.append(i * dt)             # t
        
        # Step 4: Linear regression to estimate λ₁
        # Mathematical model: ln(d(t)) = ln(d₀) + λ₁ * t
        # Slope of ln(d) vs t gives λ₁
        if len(separations) > 10:
            try:
                slope, _ = np.polyfit(times, separations, 1)  # λ₁ = slope
                lyap_estimates.append(slope)
            except:
                continue
    
    # Step 5: Statistical averaging for robust estimate
    # λ₁ = ⟨λ_local⟩ (ensemble average over multiple trajectory pairs)
    if len(lyap_estimates) > 0:
        return np.mean(lyap_estimates)
    else:
        return 0.0
    
    
def analyze_trajectory_for_chaos(trajectory_data, dt):
    """
    Quick chaos analysis for a trajectory
    
    Mathematical Framework: Multi-Modal Chaos Detection
    ==================================================
    
    This function combines multiple mathematical approaches to detect chaotic
    behavior in orbital trajectories:
    
    1. Spectral Analysis (Frequency Domain)
    2. Lyapunov Exponent Estimation (Phase Space)
    3. Statistical Characterization
    
    Parameters:
    -----------
    trajectory_data : array
        Trajectory data (n_points, 6)
    dt : float
        Time step in seconds
        
    Returns:
    --------
    chaos_summary : dict
        Summary of chaos indicators
        
    Mathematical Methods:
    --------------------
    
    A. Power Spectral Density Analysis:
       S(f) = |X(f)|²/T where X(f) = ∫ x(t)e^(-i2πft) dt
       
    B. Spectral Entropy:
       H_s = -Σᵢ pᵢ ln(pᵢ) where pᵢ = S(fᵢ)/Σⱼ S(fⱼ)
       
       Physical Meaning:
       • Low H_s: Regular/periodic motion (narrow spectrum)
       • High H_s: Chaotic motion (broad spectrum)
       
    C. Lyapunov Exponent:
       λ₁ = ⟨(1/t) ln(||δx(t)||/||δx₀||)⟩
       
       Physical Meaning:
       • λ₁ > 0: Chaotic (exponential divergence)
       • λ₁ ≈ 0: Critical/marginal behavior
       • λ₁ < 0: Stable (exponential convergence)
    """
    # Mathematical Implementation
    # ==========================
    
    # Step 1: Calculate trajectory statistics
    # Physical quantities: position and velocity magnitudes
    pos_mag = np.sqrt(np.sum(trajectory_data[:, :3]**2, axis=1))      # ||r(t)||
    vel_mag = np.sqrt(np.sum(trajectory_data[:, 3:]**2, axis=1))      # ||v(t)||
    
    # Step 2: Spectral analysis using Welch's method
    # Mathematical basis: Fourier analysis for frequency content
    freqs, psd = signal.periodogram(pos_mag, fs=1/dt)
    
    # Spectral entropy calculation
    # H_s = -Σᵢ pᵢ ln(pᵢ) where pᵢ = normalized power
    psd_normalized = psd / np.sum(psd)  # Normalize to probability distribution
    # Add small epsilon to avoid log(0)
    spectral_entropy = -np.sum(psd_normalized * np.log(psd_normalized + 1e-12))
    
    # Step 3: Lyapunov exponent estimation
    # Uses nearby trajectory method (see function documentation above)
    lyap_exp = estimate_lyapunov_exponent(trajectory_data, dt)
    
    # Step 4: Physical interpretation and classification
    # Mathematical thresholds based on orbital dynamics theory
    if lyap_exp > 1e-8:
        chaos_level = "CHAOTIC"
        lyap_time_days = 1.0 / lyap_exp / 86400    # τ = 1/λ₁ (Lyapunov time)
        predictability = f"{lyap_time_days:.1f} days"
    elif lyap_exp > -1e-8:
        chaos_level = "MARGINAL"
        predictability = "Uncertain"
    else:
        chaos_level = "STABLE"
        predictability = "Long-term"
    ##################################
    chaos_summary = {
        'lyapunov_exponent': lyap_exp,
        'chaos_level': chaos_level,
        'predictability_horizon': predictability,
        'spectral_entropy': spectral_entropy,
        'trajectory_duration_days': len(trajectory_data) * dt / 86400
    } 
    print(f"Lyapunov exponent: {chaos_summary['lyapunov_exponent']:.2e} /s")
    print(f"Chaos level: {chaos_summary['chaos_level']}")
    print(f"Predictability: {chaos_summary['predictability_horizon']}")
    print(f"Spectral entropy: {chaos_summary['spectral_entropy']:.3f}")
    
    if chaos_summary['chaos_level'] == "CHAOTIC":
        print("🔴 WARNING: Chaotic trajectory detected!")
        print("   Long-term predictions unreliable")
    elif chaos_summary['chaos_level'] == "MARGINAL":
        print("🟡 CAUTION: Marginally stable trajectory")
    else:
        print("🟢 STABLE: Predictable trajectory")
    return chaos_summary