"""
Lyapunov Exponent Analysis for Single Trajectory File
Specifically designed for analyzing trajectory files from "Databank/1950DA/tr_1e-7_6.7km/"

ACADEMIC CITATIONS AND SOURCES:
===============================

This implementation is based on established academic literature:

Primary Sources:
1. Wolf, A., Swift, J.B., Swinney, H.L., & Vastano, J.A. (1985). 
   "Determining Lyapunov exponents from a time series." 
   Physica D: Nonlinear Phenomena, 16(3), 285-317.
   DOI: 10.1016/0167-2789(85)90011-9
   [PRIMARY ALGORITHM SOURCE]

2. Rosenstein, M.T., Collins, J.J., & De Luca, C.J. (1993). 
   "A practical method for calculating largest Lyapunov exponents from 
   small data sets." Physica D: Nonlinear Phenomena, 65(1-2), 117-134.
   DOI: 10.1016/0167-2789(93)90009-P
   [PRACTICAL IMPLEMENTATION METHODS]

3. Scheeres, D.J. (2012). "Orbital Motion in Strongly Perturbed 
   Environments: Applications to Asteroid, Comet and Planetary 
   Satellite Orbiters." Springer-Praxis.
   [ORBITAL DYNAMICS THEORY]

4. Kantz, H. (1994). "A robust method to estimate the maximal Lyapunov 
   exponent of a time series." Physics Letters A, 185(1), 77-87.
   [ROBUST ESTIMATION METHODS]

Mathematical Foundations:
- Lyapunov, A.M. (1892). "The General Problem of the Stability of Motion."
- Oseledec, V.I. (1968). "A multiplicative ergodic theorem. Lyapunov 
  characteristic numbers for dynamical systems."

See references_and_citations.py for complete bibliography and additional sources.

AUTHOR: GitHub Copilot
DATE: September 2025
PURPOSE: Research and educational analysis of orbital chaos around asteroids
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.integrate import solve_ivp
import sys

# Import your existing modules
sys.dont_write_bytecode = True
import constants as C
const = C.constants()
target = C.DA1950()

def load_trajectory_file(file_path):
    """
    Load a single trajectory file
    
    Parameters:
    -----------
    file_path : str
        Path to the trajectory file
        
    Returns:
    --------
    trajectory_data : array
        Trajectory data (n_points, 6) - [x, y, z, vx, vy, vz]
    dt : float
        Time step (assumed to be 1.0 second based on your main.py)
    """
    try:
        trajectory_data = np.loadtxt(file_path, dtype=float)
        dt = 1.0  # From your main.py configuration
        print(f"Successfully loaded trajectory file: {file_path}")
        print(f"Trajectory shape: {trajectory_data.shape}")
        print(f"Duration: {trajectory_data.shape[0] * dt / 86400:.2f} days")
        return trajectory_data, dt
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None, None

def analyze_trajectory_chaos(trajectory_data, dt, omega, mu_I, CM):
    """
    Comprehensive chaos analysis for a single trajectory
    
    Parameters:
    -----------
    trajectory_data : array
        Trajectory data (n_points, 6)
    dt : float
        Time step in seconds
    omega : float
        Rotation rate of the asteroid
    mu_I : array
        Gravitational parameters of MASCONs
    CM : array
        Center of mass coordinates of MASCONs
        
    Returns:
    --------
    chaos_results : dict
        Dictionary containing all chaos analysis results
    """
    
    n_points = len(trajectory_data)
    time_array = np.arange(n_points) * dt
    
    # Initialize results dictionary
    chaos_results = {
        'file_info': {
            'n_points': n_points,
            'duration_days': n_points * dt / 86400,
            'dt': dt
        }
    }
    
    print(f"\n{'='*60}")
    print(f"CHAOS ANALYSIS FOR TRAJECTORY")
    print(f"{'='*60}")
    print(f"Points: {n_points}")
    print(f"Duration: {n_points * dt / 86400:.2f} days")
    print(f"Time step: {dt} seconds")
    
    # 1. Basic trajectory statistics
    print(f"\n{'-'*40}")
    print("TRAJECTORY STATISTICS")
    print(f"{'-'*40}")
    
    pos_mag = np.sqrt(np.sum(trajectory_data[:, :3]**2, axis=1))
    vel_mag = np.sqrt(np.sum(trajectory_data[:, 3:]**2, axis=1))
    
    chaos_results['trajectory_stats'] = {
        'position_range': [np.min(pos_mag), np.max(pos_mag)],
        'velocity_range': [np.min(vel_mag), np.max(vel_mag)],
        'position_mean_std': [np.mean(pos_mag), np.std(pos_mag)],
        'velocity_mean_std': [np.mean(vel_mag), np.std(vel_mag)]
    }
    
    print(f"Position range: {np.min(pos_mag):.3f} - {np.max(pos_mag):.3f} km")
    print(f"Velocity range: {np.min(vel_mag):.6f} - {np.max(vel_mag):.6f} km/s")
    print(f"Position: {np.mean(pos_mag):.3f} ± {np.std(pos_mag):.3f} km")
    print(f"Velocity: {np.mean(vel_mag):.6f} ± {np.std(vel_mag):.6f} km/s")
    
    # 2. Energy conservation check
    print(f"\n{'-'*40}")
    print("ENERGY CONSERVATION")
    print(f"{'-'*40}")
    
    def calc_hamiltonian_energy(state_vector, omega, mu_I, CM):
        """
        Calculate Hamiltonian energy for a single state
        
        Mathematical Formula:
        ====================
        H = T + V_eff + U
        
        where:
        • T = ½(vₓ² + vᵧ² + vᵤ²)           [Kinetic energy]
        • V_eff = -½ω²(x² + y²)           [Effective potential in rotating frame]
        • U = -Σᵢ μᵢ/||r - rᵢ||           [Gravitational potential]
        
        Physical Interpretation:
        -----------------------
        In the rotating reference frame, the Hamiltonian includes:
        1. Kinetic energy of the particle
        2. Centrifugal potential due to rotation
        3. Gravitational potential from MASCON distribution
        
        Conservation Property:
        ---------------------
        For autonomous Hamiltonian systems: dH/dt = 0
        Energy conservation is a critical test of numerical accuracy.
        """
        x, y, z, vx, vy, vz = state_vector
        
        # Gravitational potential
        U = 0.0
        for i in range(len(CM)):
            r_vec = np.array([x - CM[i,0], y - CM[i,1], z - CM[i,2]])
            r_mag = np.linalg.norm(r_vec)
            U += mu_I[i] / r_mag
        
        # Total energy in rotating frame
        kinetic = 0.5 * (vx**2 + vy**2 + vz**2)
        centrifugal = -0.5 * omega**2 * (x**2 + y**2)
        potential = -U
        
        return kinetic + centrifugal + potential
    
    # Calculate energy for each point
    energies = np.array([calc_hamiltonian_energy(point, omega, mu_I, CM) 
                        for point in trajectory_data])
    
    energy_variation = (np.max(energies) - np.min(energies)) / abs(np.mean(energies))
    
    chaos_results['energy_conservation'] = {
        'energy_range': [np.min(energies), np.max(energies)],
        'energy_mean': np.mean(energies),
        'relative_variation': energy_variation
    }
    
    print(f"Energy range: {np.min(energies):.2e} - {np.max(energies):.2e} km²/s²")
    print(f"Energy variation: {energy_variation:.2e} (relative)")
    if energy_variation < 1e-6:
        print("✓ Good energy conservation")
    else:
        print("⚠ Energy not well conserved - check integration")
    
    # 3. Spectral analysis
    print(f"\n{'-'*40}")
    print("SPECTRAL ANALYSIS")
    print(f"{'-'*40}")
    
    # Power spectral density
    freqs_pos, psd_pos = signal.periodogram(pos_mag, fs=1/dt)
    freqs_vel, psd_vel = signal.periodogram(vel_mag, fs=1/dt)
    
    # Spectral entropy (measure of complexity)
    def spectral_entropy(psd):
        psd_norm = psd / np.sum(psd)
        psd_norm = psd_norm[psd_norm > 0]
        return -np.sum(psd_norm * np.log(psd_norm))
    
    entropy_pos = spectral_entropy(psd_pos)
    entropy_vel = spectral_entropy(psd_vel)
    
    # Find dominant frequencies
    pos_peak_freq = freqs_pos[np.argmax(psd_pos[1:])+1]  # Skip DC component
    vel_peak_freq = freqs_vel[np.argmax(psd_vel[1:])+1]
    
    chaos_results['spectral_analysis'] = {
        'spectral_entropy_position': entropy_pos,
        'spectral_entropy_velocity': entropy_vel,
        'dominant_freq_position': pos_peak_freq,
        'dominant_freq_velocity': vel_peak_freq,
        'orbital_period_estimate': 1.0 / pos_peak_freq if pos_peak_freq > 0 else None
    }
    
    print(f"Spectral entropy (position): {entropy_pos:.3f}")
    print(f"Spectral entropy (velocity): {entropy_vel:.3f}")
    print(f"Dominant frequency (position): {pos_peak_freq:.2e} Hz")
    print(f"Estimated orbital period: {1.0/pos_peak_freq/3600:.2f} hours" if pos_peak_freq > 0 else "N/A")
    
    # 4. Approximate Lyapunov exponent using trajectory method
    print(f"\n{'-'*40}")
    print("LYAPUNOV EXPONENT ESTIMATION")
    print(f"{'-'*40}")
    
    # Method: Find nearby trajectories in phase space and track their separation
    def estimate_largest_lyapunov(trajectory, dt, min_separation=1e-6, max_evolution_time=None):
        """
        Estimate largest Lyapunov exponent using the method of nearby trajectories
        
        Mathematical Algorithm:
        ======================
        
        Theory:
        -------
        The largest Lyapunov exponent λ₁ quantifies the average rate of 
        exponential divergence of nearby trajectories:
        
        λ₁ = lim[T→∞] lim[δ→0] (1/T) ln(||δx(T)||/||δ₀||)
        
        where δx(t) is the separation between initially nearby trajectories.
        
        Practical Implementation:
        ------------------------
        1. Select reference points {x_ref(tᵢ)} from the trajectory
        2. For each x_ref, find nearby point x_near with ||x_ref - x_near|| ≈ δ₀
        3. Track separation evolution: d(t) = ||x_ref(t) - x_near(t)||
        4. Model exponential growth: d(t) = d₀ exp(λ₁t)
        5. Linear regression: ln(d(t)) = ln(d₀) + λ₁t → λ₁ = slope
        6. Statistical average: λ₁ = ⟨λ_local⟩
        
        Error Control:
        -------------
        • min_separation: Prevents numerical issues with log(0)
        • max_evolution_time: Limits integration to numerically stable region
        • Multiple samples: Reduces statistical uncertainty
        
        Parameters:
        -----------
        trajectory : array
            Phase space trajectory data
        dt : float  
            Time step
        min_separation : float
            Minimum allowed separation (default: 10⁻⁶)
        max_evolution_time : int
            Maximum evolution steps (default: min(1000, N/10))
            
        Returns:
        --------
        (λ_mean, λ_std) : tuple
            Mean and standard deviation of Lyapunov exponent estimates
        """
        n_points = len(trajectory)
        if max_evolution_time is None:
            max_evolution_time = min(1000, n_points // 10)  # Limit evolution time
        
        lyap_estimates = []
        
        # Sample reference points (not too many for efficiency)
        n_refs = min(100, n_points // 20)
        ref_indices = np.linspace(0, n_points - max_evolution_time - 1, n_refs, dtype=int)
        
        for ref_idx in ref_indices:
            ref_point = trajectory[ref_idx]
            
            # Find nearby points in phase space
            distances = np.sqrt(np.sum((trajectory[ref_idx:ref_idx+n_points//10] - ref_point)**2, axis=1))
            
            # Find closest point that's not the reference itself
            valid_distances = distances[1:]  # Exclude self
            if len(valid_distances) == 0 or np.min(valid_distances) < min_separation:
                continue
                
            nearest_idx = np.argmin(valid_distances) + 1  + ref_idx
            
            if nearest_idx >= n_points - max_evolution_time:
                continue
            
            # Track separation evolution
            separations = []
            times = []
            
            for i in range(min(max_evolution_time, n_points - max(ref_idx, nearest_idx))):
                if ref_idx + i < n_points and nearest_idx + i < n_points:
                    sep = np.linalg.norm(trajectory[ref_idx + i] - trajectory[nearest_idx + i])
                    if sep > min_separation:  # Avoid log(0)
                        separations.append(np.log(sep))
                        times.append(i * dt)
            
            # Fit exponential growth rate
            if len(separations) > 10:
                try:
                    slope, _ = np.polyfit(times, separations, 1)
                    lyap_estimates.append(slope)
                except:
                    continue
        
        if len(lyap_estimates) > 0:
            return np.mean(lyap_estimates), np.std(lyap_estimates)
        else:
            return 0.0, 0.0
    
    lyap_mean, lyap_std = estimate_largest_lyapunov(trajectory_data, dt)
    
    chaos_results['lyapunov_analysis'] = {
        'largest_lyapunov_estimate': lyap_mean,
        'lyapunov_uncertainty': lyap_std,
        'lyapunov_time_days': 1.0 / abs(lyap_mean) / 86400 if abs(lyap_mean) > 1e-12 else float('inf')
    }
    
    print(f"Largest Lyapunov exponent: {lyap_mean:.2e} ± {lyap_std:.2e} /s")
    
    if lyap_mean > 1e-8:  # Threshold for considering chaos
        lyap_time_days = 1.0 / lyap_mean / 86400
        print(f"✓ CHAOTIC BEHAVIOR DETECTED")
        print(f"Lyapunov time: {lyap_time_days:.1f} days")
        print(f"Predictability horizon: ~{lyap_time_days:.1f} days")
    elif lyap_mean > -1e-8:
        print("? MARGINALLY STABLE/NEUTRAL")
        print("Behavior is near the boundary between stable and chaotic")
    else:
        print("✓ STABLE BEHAVIOR")
        print("Trajectory shows stable, predictable dynamics")
    
    # 5. Poincaré section analysis
    print(f"\n{'-'*40}")
    print("POINCARÉ SECTION ANALYSIS")
    print(f"{'-'*40}")
    
    # Find crossings of x = 0 plane with positive x velocity
    poincare_points = []
    for i in range(1, len(trajectory_data)):
        if (trajectory_data[i-1, 0] * trajectory_data[i, 0] < 0 and 
            trajectory_data[i, 0] > 0):  # Crossing from negative to positive x
            # Linear interpolation to find exact crossing
            alpha = -trajectory_data[i-1, 0] / (trajectory_data[i, 0] - trajectory_data[i-1, 0])
            crossing_point = trajectory_data[i-1] + alpha * (trajectory_data[i] - trajectory_data[i-1])
            poincare_points.append(crossing_point)
    
    n_crossings = len(poincare_points)
    chaos_results['poincare_analysis'] = {
        'n_crossings': n_crossings,
        'crossing_rate': n_crossings / (n_points * dt / 86400)  # crossings per day
    }
    
    print(f"Poincaré crossings: {n_crossings}")
    print(f"Crossing rate: {n_crossings / (n_points * dt / 86400):.2f} crossings/day")
    
    # 6. Generate summary and recommendations
    print(f"\n{'='*60}")
    print("CHAOS ANALYSIS SUMMARY")
    print(f"{'='*60}")
    
    if lyap_mean > 1e-8:
        chaos_level = "CHAOTIC"
        color_code = "🔴"
    elif lyap_mean > -1e-8:
        chaos_level = "MARGINAL"
        color_code = "🟡"
    else:
        chaos_level = "STABLE"
        color_code = "🟢"
    
    print(f"{color_code} Overall Assessment: {chaos_level}")
    print(f"Energy Conservation: {'Good' if energy_variation < 1e-6 else 'Poor'}")
    print(f"Spectral Complexity: {'High' if entropy_pos > 3 else 'Low'}")
    print(f"Poincaré Crossings: {n_crossings}")
    
    chaos_results['summary'] = {
        'chaos_level': chaos_level,
        'is_chaotic': lyap_mean > 1e-8,
        'is_stable': lyap_mean < -1e-8,
        'energy_conserved': energy_variation < 1e-6,
        'high_spectral_complexity': entropy_pos > 3
    }
    
    return chaos_results

def plot_chaos_analysis(trajectory_data, chaos_results, dt, save_plots=True):
    """
    Create visualization plots for chaos analysis
    """
    time_array = np.arange(len(trajectory_data)) * dt / 86400  # Convert to days
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Chaos Analysis Results', fontsize=16)
    
    # 1. 3D Trajectory
    ax1 = axes[0, 0]
    ax1 = plt.subplot(2, 3, 1, projection='3d')
    ax1.plot(trajectory_data[:, 0], trajectory_data[:, 1], trajectory_data[:, 2], 
             'b-', alpha=0.7, linewidth=0.5)
    ax1.set_xlabel('X (km)')
    ax1.set_ylabel('Y (km)')
    ax1.set_zlabel('Z (km)')
    ax1.set_title('3D Trajectory')
    
    # 2. Position magnitude vs time
    ax2 = axes[0, 1]
    pos_mag = np.sqrt(np.sum(trajectory_data[:, :3]**2, axis=1))
    ax2.plot(time_array, pos_mag, 'r-', linewidth=1)
    ax2.set_xlabel('Time (days)')
    ax2.set_ylabel('Position (km)')
    ax2.set_title('Radial Distance vs Time')
    ax2.grid(True)
    
    # 3. Velocity magnitude vs time
    ax3 = axes[0, 2]
    vel_mag = np.sqrt(np.sum(trajectory_data[:, 3:]**2, axis=1))
    ax3.plot(time_array, vel_mag, 'g-', linewidth=1)
    ax3.set_xlabel('Time (days)')
    ax3.set_ylabel('Velocity (km/s)')
    ax3.set_title('Speed vs Time')
    ax3.grid(True)
    
    # 4. Poincaré section (y vs vy at x=0 crossings)
    ax4 = axes[1, 0]
    poincare_y = []
    poincare_vy = []
    for i in range(1, len(trajectory_data)):
        if (trajectory_data[i-1, 0] * trajectory_data[i, 0] < 0 and 
            trajectory_data[i, 0] > 0):
            alpha = -trajectory_data[i-1, 0] / (trajectory_data[i, 0] - trajectory_data[i-1, 0])
            crossing = trajectory_data[i-1] + alpha * (trajectory_data[i] - trajectory_data[i-1])
            poincare_y.append(crossing[1])
            poincare_vy.append(crossing[4])
    
    if len(poincare_y) > 0:
        ax4.scatter(poincare_y, poincare_vy, c='red', s=1, alpha=0.7)
        ax4.set_xlabel('Y (km)')
        ax4.set_ylabel('VY (km/s)')
        ax4.set_title(f'Poincaré Section (x=0)\n{len(poincare_y)} points')
        ax4.grid(True)
    else:
        ax4.text(0.5, 0.5, 'No Poincaré crossings found', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Poincaré Section (x=0)')
    
    # 5. Power spectral density
    ax5 = axes[1, 1]
    freqs, psd = signal.periodogram(pos_mag, fs=1/dt)
    # Plot only non-zero frequencies and limit to reasonable range
    valid_freqs = freqs[freqs > 0]
    valid_psd = psd[freqs > 0]
    ax5.loglog(valid_freqs, valid_psd, 'b-', linewidth=1)
    ax5.set_xlabel('Frequency (Hz)')
    ax5.set_ylabel('Power Spectral Density')
    ax5.set_title('PSD of Radial Distance')
    ax5.grid(True)
    
    # 6. Phase space projection (x vs vx)
    ax6 = axes[1, 2]
    # Subsample for visibility
    subsample = slice(None, None, max(1, len(trajectory_data) // 5000))
    ax6.plot(trajectory_data[subsample, 0], trajectory_data[subsample, 3], 
             'purple', alpha=0.6, linewidth=0.5)
    ax6.set_xlabel('X (km)')
    ax6.set_ylabel('VX (km/s)')
    ax6.set_title('Phase Space (X-VX)')
    ax6.grid(True)
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('chaos_analysis_plots.png', dpi=300, bbox_inches='tight')
        print(f"\nPlots saved as 'chaos_analysis_plots.png'")
    
    plt.show()

def analyze_single_trajectory_file(file_path):
    """
    Main function to analyze a single trajectory file for chaos
    """
    # Load asteroid parameters
    omega = 2.0 * np.pi / (target.spin * 3600.0)  # Rotation rate
    
    # Load MASCON data
    CM_Path = '1950DA_Prograde_CM.in'
    mu_Path = '1950DA_Prograde_mu.in'
    
    try:
        CM = np.loadtxt(CM_Path, delimiter=' ', dtype=float)
        mu_I = np.loadtxt(mu_Path, delimiter=' ')
        print(f"Loaded {len(CM)} MASCONs")
        print(f"Total gravitational parameter: {np.sum(mu_I):.2e}")
    except:
        print("Warning: Could not load MASCON data. Some analyses will be limited.")
        CM = np.array([[0, 0, 0]])  # Dummy data
        mu_I = np.array([1.0])      # Dummy data
    
    # Load and analyze trajectory
    trajectory_data, dt = load_trajectory_file(file_path)
    
    if trajectory_data is None:
        return None
    
    # Perform chaos analysis
    chaos_results = analyze_trajectory_chaos(trajectory_data, dt, omega, mu_I, CM)
    
    # Create plots
    plot_chaos_analysis(trajectory_data, chaos_results, dt)
    
    return chaos_results

if __name__ == "__main__":
    # Example usage - analyze a specific file
    folder = "Databank/1950DA/tr_1e-7_6.7km/"
    
    # List available files in the directory
    if os.path.exists(folder):
        files = [f for f in os.listdir(folder) if f.endswith('.dat')]
        print(f"Found {len(files)} trajectory files in {folder}")
        
        if len(files) > 0:
            # Analyze the first file as an example
            example_file = os.path.join(folder, files[0])
            print(f"\nAnalyzing: {example_file}")
            
            results = analyze_single_trajectory_file(example_file)
            
            # Option to analyze more files
            if len(files) > 1:
                print(f"\nOther available files:")
                for i, file in enumerate(files[1:6]):  # Show up to 5 more
                    print(f"  {i+2}. {file}")
                if len(files) > 6:
                    print(f"  ... and {len(files)-6} more files")
        else:
            print(f"No .dat files found in {folder}")
    else:
        print(f"Directory {folder} not found")
        
    # You can also analyze a specific file by uncommenting:
    specific_file = "Databank/1950DA/tr_1e-7_6.7km/TR-S0-H1e-07Yi1.7.dat"
    results = analyze_single_trajectory_file(specific_file)
