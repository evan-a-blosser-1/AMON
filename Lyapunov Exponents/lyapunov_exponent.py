"""
Lyapunov Exponent Calculator for Orbital Trajectories
Specifically designed for asteroid orbital dynamics
"""

import numpy as np
from scipy.integrate import solve_ivp
from numba import njit
import matplotlib.pyplot as plt

class LyapunovCalculator:
    """
    Calculate Lyapunov exponents for orbital trajectories using the method
    of nearby trajectory separation.
    """
    
    def __init__(self, eom_func, n_dim=6):
        """
        Initialize Lyapunov calculator
        
        Parameters:
        -----------
        eom_func : callable
            Equations of motion function (t, state, *args) -> derivatives
        n_dim : int
            Dimension of the state space (default 6 for 3D orbital mechanics)
        """
        self.eom_func = eom_func
        self.n_dim = n_dim
        
    def variational_equations(self, t, state_extended, *args):
        """
        Combined equations for the reference trajectory and variational equations
        
        Parameters:
        -----------
        t : float
            Time
        state_extended : array
            Extended state vector [x0, x1, ..., xn, dx0, dx1, ..., dxn]
        *args : tuple
            Additional arguments for the EOM function
            
        Returns:
        --------
        derivatives : array
            Time derivatives of the extended state
        """
        n = self.n_dim
        
        # Split state into reference trajectory and perturbation
        x_ref = state_extended[:n]
        dx = state_extended[n:].reshape((n, n))
        
        # Calculate reference trajectory derivatives
        dxdt_ref = np.array(self.eom_func(t, x_ref, *args))
        
        # Calculate Jacobian numerically
        eps = 1e-8
        jacobian = np.zeros((n, n))
        
        for i in range(n):
            x_plus = x_ref.copy()
            x_minus = x_ref.copy()
            x_plus[i] += eps
            x_minus[i] -= eps
            
            f_plus = np.array(self.eom_func(t, x_plus, *args))
            f_minus = np.array(self.eom_func(t, x_minus, *args))
            
            jacobian[:, i] = (f_plus - f_minus) / (2 * eps)
        
        # Variational equation: d(dx)/dt = J * dx
        ddxdt = jacobian @ dx
        
        # Combine derivatives
        derivatives = np.concatenate([dxdt_ref, ddxdt.flatten()])
        
        return derivatives
    
    def calculate_lyapunov_spectrum(self, initial_state, t_span, dt, 
                                   renorm_interval=100, *args):
        """
        Calculate the full Lyapunov spectrum using Gram-Schmidt orthogonalization
        
        Parameters:
        -----------
        initial_state : array
            Initial conditions for the reference trajectory
        t_span : tuple
            (t_start, t_end) integration time span
        dt : float
            Time step
        renorm_interval : int
            Number of integration steps between renormalizations
        *args : tuple
            Additional arguments for the EOM function
            
        Returns:
        --------
        lyapunov_exponents : array
            Lyapunov exponents sorted in descending order
        time_evolution : array
            Time evolution of the exponents
        """
        n = self.n_dim
        
        # Initialize perturbation matrix (identity)
        dx_init = np.eye(n).flatten()
        
        # Extended initial state
        state_init = np.concatenate([initial_state, dx_init])
        
        # Time array
        t_eval = np.arange(t_span[0], t_span[1], dt)
        n_steps = len(t_eval)
        
        # Storage for Lyapunov exponents
        lyap_sums = np.zeros(n)
        lyap_evolution = np.zeros((n_steps // renorm_interval, n))
        
        current_time = t_span[0]
        current_state = state_init.copy()
        
        step_count = 0
        renorm_count = 0
        
        for i in range(1, n_steps):
            # Integrate for one time step
            sol = solve_ivp(
                self.variational_equations,
                [current_time, current_time + dt],
                current_state,
                args=args,
                method='DOP853',
                rtol=1e-10,
                atol=1e-12,
                dense_output=False
            )
            
            current_state = sol.y[:, -1]
            current_time += dt
            step_count += 1
            
            # Renormalization step
            if step_count % renorm_interval == 0:
                # Extract perturbation matrix
                dx_matrix = current_state[n:].reshape((n, n))
                
                # QR decomposition for Gram-Schmidt orthogonalization
                Q, R = np.linalg.qr(dx_matrix)
                
                # Update Lyapunov sum
                for j in range(n):
                    lyap_sums[j] += np.log(abs(R[j, j]))
                
                # Store current estimates
                current_lyaps = lyap_sums / (current_time - t_span[0])
                lyap_evolution[renorm_count] = current_lyaps.copy()
                
                # Reset perturbation matrix to orthonormal
                current_state[n:] = Q.flatten()
                
                renorm_count += 1
        
        # Final Lyapunov exponents
        total_time = t_span[1] - t_span[0]
        lyapunov_exponents = lyap_sums / total_time
        
        # Sort in descending order
        sorted_indices = np.argsort(lyapunov_exponents)[::-1]
        lyapunov_exponents = lyapunov_exponents[sorted_indices]
        
        return lyapunov_exponents, lyap_evolution[:renorm_count]
    
    def largest_lyapunov_exponent(self, initial_state, t_span, dt, 
                                  separation_threshold=1e-6, *args):
        """
        Calculate only the largest Lyapunov exponent using trajectory separation
        (Faster method for just the largest exponent)
        
        Parameters:
        -----------
        initial_state : array
            Initial conditions for the reference trajectory
        t_span : tuple
            (t_start, t_end) integration time span
        dt : float
            Time step
        separation_threshold : float
            Threshold for trajectory separation
        *args : tuple
            Additional arguments for the EOM function
            
        Returns:
        --------
        largest_lyapunov : float
            Largest Lyapunov exponent
        """
        # Small random perturbation
        perturbation = np.random.normal(0, 1e-12, len(initial_state))
        perturbed_state = initial_state + perturbation
        
        # Time array
        t_eval = np.arange(t_span[0], t_span[1], dt)
        
        lyap_sum = 0.0
        rescale_count = 0
        
        # Initial separation
        d0 = np.linalg.norm(perturbation)
        
        for i in range(1, len(t_eval)):
            # Integrate both trajectories
            sol_ref = solve_ivp(
                self.eom_func,
                [t_eval[i-1], t_eval[i]],
                initial_state,
                args=args,
                method='DOP853',
                rtol=1e-10,
                atol=1e-12
            )
            
            sol_pert = solve_ivp(
                self.eom_func,
                [t_eval[i-1], t_eval[i]],
                perturbed_state,
                args=args,
                method='DOP853',
                rtol=1e-10,
                atol=1e-12
            )
            
            # Update states
            initial_state = sol_ref.y[:, -1]
            perturbed_state = sol_pert.y[:, -1]
            
            # Calculate separation
            separation = perturbed_state - initial_state
            d1 = np.linalg.norm(separation)
            
            # Check if rescaling is needed
            if d1 > separation_threshold:
                # Add to Lyapunov sum
                lyap_sum += np.log(d1 / d0)
                rescale_count += 1
                
                # Rescale perturbation
                perturbed_state = initial_state + (separation / d1) * d0
                
        # Calculate largest Lyapunov exponent
        total_time = t_span[1] - t_span[0]
        largest_lyapunov = lyap_sum / total_time
        
        return largest_lyapunov

def analyze_orbital_chaos(trajectory_data, time_data, dt):
    """
    Analyze existing trajectory data for chaotic behavior
    
    Parameters:
    -----------
    trajectory_data : array
        Trajectory data (n_points, 6) - [x, y, z, vx, vy, vz]
    time_data : array
        Time array corresponding to trajectory data
    dt : float
        Time step
        
    Returns:
    --------
    chaos_indicators : dict
        Dictionary containing various chaos indicators
    """
    
    def trajectory_eom(t, state):
        """Dummy EOM - not used for existing data analysis"""
        return np.zeros_like(state)
    
    # Calculate position and velocity magnitudes
    pos_mag = np.sqrt(np.sum(trajectory_data[:, :3]**2, axis=1))
    vel_mag = np.sqrt(np.sum(trajectory_data[:, 3:]**2, axis=1))
    
    # Simple chaos indicators
    chaos_indicators = {}
    
    # 1. Power spectral density analysis
    from scipy import signal
    freqs, psd_pos = signal.periodogram(pos_mag, fs=1/dt)
    freqs, psd_vel = signal.periodogram(vel_mag, fs=1/dt)
    
    # Spectral entropy (measure of complexity)
    def spectral_entropy(psd):
        psd_norm = psd / np.sum(psd)
        psd_norm = psd_norm[psd_norm > 0]  # Remove zeros
        return -np.sum(psd_norm * np.log(psd_norm))
    
    chaos_indicators['spectral_entropy_pos'] = spectral_entropy(psd_pos)
    chaos_indicators['spectral_entropy_vel'] = spectral_entropy(psd_vel)
    
    # 2. Approximate Lyapunov exponent from trajectory divergence
    # (Rough estimate without full variational equations)
    separations = []
    n_neighbors = min(100, len(trajectory_data) // 10)
    
    for i in range(0, len(trajectory_data) - n_neighbors, n_neighbors):
        ref_point = trajectory_data[i]
        
        # Find nearby points
        distances = np.sqrt(np.sum((trajectory_data[i:i+n_neighbors] - ref_point)**2, axis=1))
        nearest_idx = np.argmin(distances[1:]) + 1  # Exclude self
        
        if distances[nearest_idx] > 0:
            # Track separation evolution
            sep_evolution = []
            max_len = min(50, len(trajectory_data) - i - nearest_idx)
            
            for j in range(max_len):
                if i + j < len(trajectory_data) and i + nearest_idx + j < len(trajectory_data):
                    sep = np.linalg.norm(trajectory_data[i + j] - trajectory_data[i + nearest_idx + j])
                    if sep > 0:
                        sep_evolution.append(np.log(sep))
            
            if len(sep_evolution) > 10:
                # Linear fit to log(separation) vs time
                time_subset = np.arange(len(sep_evolution)) * dt
                slope, _ = np.polyfit(time_subset, sep_evolution, 1)
                separations.append(slope)
    
    chaos_indicators['approx_largest_lyapunov'] = np.mean(separations) if separations else 0.0
    chaos_indicators['lyapunov_std'] = np.std(separations) if separations else 0.0
    
    # 3. Correlation dimension (simplified)
    def correlation_dimension(data, max_r=None, n_points=1000):
        if max_r is None:
            max_r = np.std(data) * 2
        
        # Sample points for efficiency
        indices = np.random.choice(len(data), min(n_points, len(data)), replace=False)
        sample_data = data[indices]
        
        r_values = np.logspace(-3, np.log10(max_r), 20)
        correlations = []
        
        for r in r_values:
            count = 0
            total = 0
            for i in range(len(sample_data)):
                for j in range(i+1, len(sample_data)):
                    total += 1
                    if np.linalg.norm(sample_data[i] - sample_data[j]) < r:
                        count += 1
            correlations.append(count / total if total > 0 else 0)
        
        # Fit line to log-log plot
        valid_indices = np.where(np.array(correlations) > 0)[0]
        if len(valid_indices) > 5:
            log_r = np.log(r_values[valid_indices])
            log_c = np.log(np.array(correlations)[valid_indices])
            slope, _ = np.polyfit(log_r, log_c, 1)
            return slope
        return 0.0
    
    chaos_indicators['correlation_dimension'] = correlation_dimension(trajectory_data)
    
    return chaos_indicators

# Example usage function for your existing code
def integrate_with_lyapunov_analysis():
    """
    Example of how to integrate Lyapunov analysis with your existing orbital code
    """
    print("Example integration with your existing AMON code:")
    print("""
    # In your main.py or a new analysis script:
    from lyapunov_exponent import LyapunovCalculator, analyze_orbital_chaos
    
    # Create Lyapunov calculator
    lyap_calc = LyapunovCalculator(EOM_MASCON)
    
    # For a specific trajectory, calculate Lyapunov exponents
    initial_conditions = [0.0, 1.0, 0.0, x_dot, 0.0, 0.0]  # Your IC
    t_span = (0, 86400 * 30)  # 30 days
    dt = 1.0  # 1 second
    
    # Calculate full Lyapunov spectrum
    lyap_spectrum, evolution = lyap_calc.calculate_lyapunov_spectrum(
        initial_conditions, t_span, dt, args=(CM, mu_I, omega, Ham)
    )
    
    print(f"Largest Lyapunov exponent: {lyap_spectrum[0]:.2e} /s")
    if lyap_spectrum[0] > 0:
        print("Orbit shows chaotic behavior")
        chaos_time = 1.0 / lyap_spectrum[0]  # Lyapunov time
        print(f"Characteristic chaos timescale: {chaos_time/86400:.1f} days")
    else:
        print("Orbit appears stable")
    
    # For existing trajectory data analysis
    # Load your trajectory file
    trajectory = np.loadtxt("your_trajectory_file.dat")
    time_array = np.arange(len(trajectory)) * dt
    
    chaos_indicators = analyze_orbital_chaos(trajectory, time_array, dt)
    print(f"Spectral entropy: {chaos_indicators['spectral_entropy_pos']:.3f}")
    print(f"Approximate Lyapunov: {chaos_indicators['approx_largest_lyapunov']:.2e}")
    """)

if __name__ == "__main__":
    integrate_with_lyapunov_analysis()
