"""
Mathematical Foundation for Lyapunov Exponent Calculation in Orbital Dynamics
==============================================================================

ACADEMIC SOURCES AND CITATIONS:
===============================

This mathematical documentation is based on the following academic sources:

FOUNDATIONAL THEORY:
1. Lyapunov, A.M. (1892). "The General Problem of the Stability of Motion."
2. Oseledec, V.I. (1968). "A multiplicative ergodic theorem."
3. Wolf, A. et al. (1985). "Determining Lyapunov exponents from a time series."
   Physica D: Nonlinear Phenomena, 16(3), 285-317.

PRACTICAL METHODS:
4. Rosenstein, M.T. et al. (1993). "A practical method for calculating largest
   Lyapunov exponents from small data sets." Physica D, 65(1-2), 117-134.
5. Kantz, H. (1994). "A robust method to estimate the maximal Lyapunov exponent."
   Physics Letters A, 185(1), 77-87.

ORBITAL MECHANICS:
6. Scheeres, D.J. (2012). "Orbital Motion in Strongly Perturbed Environments."
7. Murray, C.D. & Dermott, S.F. (1999). "Solar System Dynamics."
8. Szebehely, V. (1967). "Theory of Orbits: The Restricted Problem of Three Bodies."

DYNAMICAL SYSTEMS:
9. Strogatz, S.H. (2014). "Nonlinear Dynamics and Chaos." 2nd Edition.
10. Wiggins, S. (2003). "Introduction to Applied Nonlinear Dynamical Systems."

See references_and_citations.py for complete bibliography with DOIs and details.

Author: GitHub Copilot
Date: September 2025
Context: Asteroid orbital dynamics analysis (AMON project)
"""

import numpy as np
import matplotlib.pyplot as plt

class LyapunovMathematics:
    """
    Mathematical documentation and implementation of Lyapunov exponent theory
    for orbital dynamics around irregular gravitational bodies.
    """
    
    def __init__(self):
        """Initialize mathematical documentation"""
        pass
    
    def theoretical_background(self):
        """
        Theoretical Background
        ====================
        
        Lyapunov exponents quantify the rate of exponential divergence of nearby
        trajectories in phase space. For a dynamical system:
        
        dx/dt = f(x,t)
        
        where x ∈ ℝⁿ is the state vector, the largest Lyapunov exponent λ₁ is defined as:
        
        λ₁ = lim[t→∞] lim[δx₀→0] (1/t) ln(||δx(t)||/||δx₀||)
        
        where δx(t) is the separation between two initially nearby trajectories.
        
        Physical Interpretation:
        -----------------------
        • λ₁ > 0: Chaotic behavior (exponential divergence)
        • λ₁ = 0: Marginal stability (neutral behavior)  
        • λ₁ < 0: Stable behavior (exponential convergence)
        
        The Lyapunov time τ = 1/|λ₁| represents the characteristic timescale
        over which predictions become unreliable in chaotic systems.
        """
        print(self.theoretical_background.__doc__)
    
    def orbital_dynamics_equations(self):
        """
        Orbital Dynamics in Rotating Frame
        =================================
        
        For a particle orbiting an asteroid in a rotating reference frame,
        the equations of motion are:
        
        d²r/dt² = -∇U(r) - 2Ω × dr/dt - Ω × (Ω × r)
        
        where:
        • r = [x, y, z]ᵀ is the position vector
        • U(r) is the gravitational potential
        • Ω = [0, 0, ω]ᵀ is the angular velocity vector
        • ω is the rotation rate of the asteroid
        
        State Vector:
        ------------
        The 6D state vector is: x = [x, y, z, vₓ, vᵧ, vᵤ]ᵀ
        
        Equations of Motion:
        ------------------
        dx/dt = vₓ
        dy/dt = vᵧ  
        dz/dt = vᵤ
        dvₓ/dt = ω²x + 2ωvᵧ + ∂U/∂x
        dvᵧ/dt = ω²y - 2ωvₓ + ∂U/∂y
        dvᵤ/dt = ∂U/∂z
        
        Gravitational Potential (MASCON Model):
        -------------------------------------
        U(r) = -∑ᵢ μᵢ/||r - rᵢ||
        
        where μᵢ and rᵢ are the gravitational parameter and position
        of the i-th MASCON (point mass).
        """
        print(self.orbital_dynamics_equations.__doc__)
    
    def lyapunov_calculation_methods(self):
        """
        Lyapunov Exponent Calculation Methods
        ===================================
        
        Method 1: Variational Equations (Full Method)
        --------------------------------------------
        The most rigorous approach involves integrating the linearized system:
        
        d/dt[x(t)]     = [f(x(t),t)]
            [δx(t)]       [J(x(t),t)·δx(t)]
        
        where J is the Jacobian matrix:
        J_{ij} = ∂fᵢ/∂xⱼ
        
        Gram-Schmidt orthogonalization is performed periodically to prevent
        numerical overflow:
        
        λᵢ = (1/T) ∑ₖ ln(Rᵢᵢ⁽ᵏ⁾)
        
        where Rᵢᵢ⁽ᵏ⁾ are diagonal elements from QR decomposition at time step k.
        
        Method 2: Nearby Trajectory Method (Practical Implementation)
        -----------------------------------------------------------
        Used in our implementation for existing trajectory data:
        
        1. Find nearby points in phase space:
           d₀ = ||x(t₀) - x'(t₀)|| where x' is a nearby trajectory point
        
        2. Track separation evolution:
           d(t) = ||x(t) - x'(t)||
        
        3. Calculate local Lyapunov exponent:
           λ_local = (1/Δt) ln(d(t + Δt)/d(t))
        
        4. Average over multiple pairs and time intervals:
           λ₁ ≈ ⟨λ_local⟩
        
        Implementation Details:
        ----------------------
        • Minimum separation threshold: δ_min = 10⁻⁶ to 10⁻⁸
        • Maximum evolution time: limited to avoid numerical issues
        • Multiple reference points: statistical averaging improves accuracy
        """
        print(self.lyapunov_calculation_methods.__doc__)
    
    def jacobian_matrix_orbital(self):
        """
        Jacobian Matrix for Orbital Dynamics
        ===================================
        
        For the orbital system dx/dt = f(x), the Jacobian J = ∂f/∂x is:
        
        J = [0₃ₓ₃    I₃ₓ₃]
            [∇²U    2ωΣ + ω²I₂]
        
        where:
        • 0₃ₓ₃ is the 3×3 zero matrix
        • I₃ₓ₃ is the 3×3 identity matrix
        • ∇²U is the Hessian of the gravitational potential
        • Σ is the skew-symmetric matrix for Coriolis terms
        
        Hessian Matrix Elements:
        ----------------------
        For MASCON potential U = -∑ᵢ μᵢ/rᵢ where rᵢ = ||r - rᵢ||:
        
        ∂²U/∂x² = ∑ᵢ μᵢ[3(x-xᵢ)²/rᵢ⁵ - 1/rᵢ³]
        ∂²U/∂y² = ∑ᵢ μᵢ[3(y-yᵢ)²/rᵢ⁵ - 1/rᵢ³]
        ∂²U/∂z² = ∑ᵢ μᵢ[3(z-zᵢ)²/rᵢ⁵ - 1/rᵢ³]
        ∂²U/∂x∂y = ∑ᵢ μᵢ[3(x-xᵢ)(y-yᵢ)/rᵢ⁵]
        ∂²U/∂x∂z = ∑ᵢ μᵢ[3(x-xᵢ)(z-zᵢ)/rᵢ⁵]
        ∂²U/∂y∂z = ∑ᵢ μᵢ[3(y-yᵢ)(z-zᵢ)/rᵢ⁵]
        
        Coriolis Matrix:
        ---------------
        Σ = [0  -1   0]
            [1   0   0]
            [0   0   0]
        """
        print(self.jacobian_matrix_orbital.__doc__)
    
    def numerical_implementation(self):
        """
        Numerical Implementation Details
        ==============================
        
        Algorithm: Nearby Trajectory Method
        ----------------------------------
        
        Input: Trajectory data T = {x(tᵢ)}ᵢ₌₀ᴺ, time step Δt
        
        1. Initialize:
           - Reference points: sample M points from trajectory
           - Lyapunov estimates: L = []
        
        2. For each reference point xᵣₑf(tᵣ):
           a) Find nearby point x_near(tₙ) such that:
              ||xᵣₑf(tᵣ) - x_near(tₙ)|| = min{||xᵣₑf(tᵣ) - x(tᵢ)||} 
              subject to |tᵣ - tᵢ| < τ_search and ||·|| > δ_min
           
           b) Track separation evolution:
              For k = 0, 1, ..., K_max:
                 d(k) = ||x(tᵣ + kΔt) - x(tₙ + kΔt)||
                 if d(k) > δ_min: s(k) = ln(d(k))
           
           c) Linear regression:
              λ_local = slope of s(k) vs (kΔt)
              if regression quality > threshold: L.append(λ_local)
        
        3. Statistical processing:
           λ₁ = mean(L)
           σ_λ = std(L)
        
        Parameters Used:
        ---------------
        • δ_min = 10⁻⁶ (minimum separation threshold)
        • M ≤ 100 (number of reference points)
        • K_max ≤ 1000 (maximum evolution steps)
        • τ_search = N/10 (search window for nearby points)
        
        Error Sources and Mitigation:
        ----------------------------
        1. Finite precision: Use double precision arithmetic
        2. Trajectory discretization: Ensure Δt << orbital period
        3. Limited evolution time: Balance accuracy vs numerical stability
        4. Statistical sampling: Average over multiple trajectory pairs
        """
        print(self.numerical_implementation.__doc__)
    
    def spectral_analysis_theory(self):
        """
        Spectral Analysis for Chaos Detection
        ===================================
        
        Power Spectral Density:
        ----------------------
        For a time series x(t), the power spectral density is:
        
        S(f) = |X(f)|²/T
        
        where X(f) is the Fourier transform and T is the observation time.
        
        Spectral Entropy:
        ----------------
        A measure of spectral complexity:
        
        H_s = -∑ᵢ pᵢ ln(pᵢ)
        
        where pᵢ = S(fᵢ)/∑ⱼS(fⱼ) is the normalized power at frequency fᵢ.
        
        Interpretation:
        • Low H_s: Periodic/quasi-periodic motion (few dominant frequencies)
        • High H_s: Complex/chaotic motion (broad frequency spectrum)
        
        Relationship to Chaos:
        ---------------------
        Chaotic systems typically exhibit:
        • Broad-band power spectra
        • High spectral entropy
        • Absence of discrete spectral lines
        
        Implementation:
        --------------
        Using Welch's method (scipy.signal.periodogram) for robust PSD estimation
        with appropriate windowing to reduce spectral leakage.
        """
        print(self.spectral_analysis_theory.__doc__)
    
    def energy_conservation_check(self):
        """
        Energy Conservation Analysis
        ===========================
        
        Hamiltonian in Rotating Frame:
        -----------------------------
        H = ½(vₓ² + vᵧ² + vᵤ²) - ½ω²(x² + y²) + U(x,y,z)
        
        where:
        • Kinetic energy: T = ½(vₓ² + vᵧ² + vᵤ²)
        • Centrifugal potential: V_cf = -½ω²(x² + y²)
        • Gravitational potential: U(r) = -∑ᵢ μᵢ/||r - rᵢ||
        
        Conservation Test:
        -----------------
        For accurate numerical integration, energy should be conserved:
        
        ΔH = (H_max - H_min)/|H_mean| << 1
        
        Typical thresholds:
        • ΔH < 10⁻⁶: Excellent conservation
        • ΔH < 10⁻³: Acceptable conservation  
        • ΔH > 10⁻³: Poor conservation (check integrator settings)
        
        Physical Significance:
        ---------------------
        Energy conservation violations indicate:
        1. Insufficient numerical precision
        2. Too large integration time step
        3. Numerical instabilities near singularities
        
        Impact on Lyapunov Calculation:
        ------------------------------
        Poor energy conservation can lead to:
        • Artificial trajectory divergence
        • Overestimation of Lyapunov exponents
        • False positive chaos detection
        """
        print(self.energy_conservation_check.__doc__)
    
    def poincare_section_analysis(self):
        """
        Poincaré Section Analysis
        ========================
        
        Definition:
        ----------
        A Poincaré section is the intersection of a trajectory with a 
        lower-dimensional subspace (hyperplane) in phase space.
        
        Implementation:
        --------------
        We use the hyperplane x = 0 with the condition vₓ > 0 (crossing
        from negative to positive x).
        
        Crossing Detection:
        ------------------
        For consecutive points (x₁, x₂) where x₁ < 0 and x₂ > 0:
        
        Linear interpolation to find exact crossing:
        α = -x₁/(x₂ - x₁)
        x_crossing = x₁ + α(x₂ - x₁)
        
        Chaos Indicators from Poincaré Sections:
        ---------------------------------------
        1. Regular motion: Discrete points or simple curves
        2. Quasi-periodic: Smooth curves (tori intersections)  
        3. Chaotic motion: Scattered points, fractal structure
        
        Statistical Measures:
        -------------------
        • Number of crossings: N_cross
        • Crossing rate: N_cross/(total_time)
        • Spatial distribution of crossing points
        
        Relationship to Lyapunov Exponents:
        ----------------------------------
        • Regular sections ↔ λ₁ ≤ 0
        • Fractal/scattered sections ↔ λ₁ > 0
        """
        print(self.poincare_section_analysis.__doc__)
    
    def validation_and_benchmarks(self):
        """
        Validation and Benchmarking
        ===========================
        
        Known Test Cases:
        ----------------
        1. Harmonic Oscillator:
           x'' + ω₀²x = 0
           Expected: λ₁ = 0 (neutral stability)
        
        2. Lorenz System:
           dx/dt = σ(y - x)
           dy/dt = x(ρ - z) - y  
           dz/dt = xy - βz
           Expected: λ₁ ≈ 0.9 for standard parameters
        
        3. Hénon Map:
           x_{n+1} = 1 - ax_n² + y_n
           y_{n+1} = bx_n
           Expected: λ₁ ≈ 0.42 for a=1.4, b=0.3
        
        Validation Criteria:
        -------------------
        1. Convergence: λ₁ should converge as evolution time increases
        2. Statistical consistency: Multiple runs should give consistent results
        3. Parameter sensitivity: Results should be robust to numerical parameters
        
        Accuracy Assessment:
        -------------------
        • Compare with analytical results (where available)
        • Cross-validate with different methods
        • Test sensitivity to numerical parameters
        
        Typical Accuracy:
        ----------------
        • Order of magnitude: Usually reliable
        • Factor of 2-3: Expected for complex systems
        • Sign (±): Most robust indicator of stability/chaos
        """
        print(self.validation_and_benchmarks.__doc__)
    
    def interpretation_guidelines(self):
        """
        Physical Interpretation Guidelines
        =================================
        
        Lyapunov Exponent Ranges for Orbital Systems:
        --------------------------------------------
        
        Highly Stable (λ₁ < -10⁻⁶ s⁻¹):
        • Well-behaved elliptical orbits
        • Predictable for geological timescales
        • Suitable for long-term mission planning
        
        Stable (−10⁻⁶ < λ₁ < −10⁻⁹ s⁻¹):
        • Mildly perturbed orbits
        • Predictable for years to decades
        • Requires periodic orbit corrections
        
        Marginally Stable (−10⁻⁹ < λ₁ < 10⁻⁹ s⁻¹):
        • Near resonances or equilibrium points
        • Boundary between order and chaos
        • High sensitivity to perturbations
        
        Weakly Chaotic (10⁻⁹ < λ₁ < 10⁻⁶ s⁻¹):
        • Chaotic but slowly diverging
        • Predictable for days to months
        • Common in asteroid proximity operations
        
        Strongly Chaotic (λ₁ > 10⁻⁶ s⁻¹):
        • Rapidly diverging trajectories  
        • Predictable for hours to days only
        • Avoid for spacecraft operations
        
        Mission Planning Implications:
        -----------------------------
        
        Lyapunov Time τ = 1/|λ₁|:
        • τ > 1 year: Excellent for long-term missions
        • 1 month < τ < 1 year: Good for medium-term operations
        • 1 day < τ < 1 month: Requires frequent updates
        • τ < 1 day: Not suitable for autonomous operations
        
        Orbit Categories:
        ----------------
        
        Safe Orbits (λ₁ < 0):
        • Bound, stable motion
        • Predictable evolution
        • Suitable for science operations
        
        Risky Orbits (λ₁ ≈ 0):
        • Near escape or collision boundaries
        • Sensitive to perturbations
        • Requires careful monitoring
        
        Dangerous Orbits (λ₁ > 0):
        • Unpredictable long-term evolution
        • High collision/escape probability
        • Avoid for crewed missions
        """
        print(self.interpretation_guidelines.__doc__)

def demonstrate_calculation():
    """
    Demonstrate Lyapunov calculation with a simple example
    """
    print("\n" + "="*60)
    print("MATHEMATICAL DEMONSTRATION")
    print("="*60)
    
    # Simple example: tracking separation growth
    print("\nExample: Exponential Separation Growth")
    print("-" * 40)
    
    # Simulated data
    t = np.linspace(0, 10, 100)
    lambda_true = 0.1  # True Lyapunov exponent
    
    # Two nearby trajectories
    d0 = 1e-6  # Initial separation
    separation = d0 * np.exp(lambda_true * t)
    
    # Add some noise to make it realistic
    noise = 1 + 0.1 * np.random.randn(len(t))
    separation_noisy = separation * noise
    
    # Calculate Lyapunov exponent from data
    log_sep = np.log(separation_noisy)
    lambda_estimated, _ = np.polyfit(t, log_sep, 1)
    
    print(f"True Lyapunov exponent: {lambda_true:.3f}")
    print(f"Estimated Lyapunov exponent: {lambda_estimated:.3f}")
    print(f"Error: {abs(lambda_estimated - lambda_true):.3f}")
    
    # Plot
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.semilogy(t, separation, 'b-', label='True separation')
    plt.semilogy(t, separation_noisy, 'r.', alpha=0.7, label='Noisy data')
    plt.xlabel('Time')
    plt.ylabel('Separation')
    plt.title('Trajectory Separation vs Time')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(t, log_sep, 'r.', alpha=0.7, label='Data')
    plt.plot(t, lambda_estimated * t + np.log(d0), 'b-', 
             label=f'Fit: λ = {lambda_estimated:.3f}')
    plt.xlabel('Time')
    plt.ylabel('ln(separation)')
    plt.title('Linear Fit to Log-Separation')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('lyapunov_demonstration.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\nDemonstration plot saved as 'lyapunov_demonstration.png'")

if __name__ == "__main__":
    # Create documentation instance
    math_doc = LyapunovMathematics()
    
    # Print all mathematical sections
    print("MATHEMATICAL FOUNDATION FOR LYAPUNOV EXPONENT CALCULATION")
    print("=" * 80)
    
    math_doc.theoretical_background()
    math_doc.orbital_dynamics_equations()
    math_doc.lyapunov_calculation_methods()
    math_doc.jacobian_matrix_orbital()
    math_doc.numerical_implementation()
    math_doc.spectral_analysis_theory()
    math_doc.energy_conservation_check()
    math_doc.poincare_section_analysis()
    math_doc.validation_and_benchmarks()
    math_doc.interpretation_guidelines()
    
    # Run demonstration
    demonstrate_calculation()
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("""
    The Lyapunov exponent calculation implemented in the AMON orbital
    analysis tools is based on rigorous dynamical systems theory adapted
    for practical application to asteroid orbital mechanics.
    
    Key Mathematical Components:
    1. Hamiltonian dynamics in rotating reference frame
    2. Nearby trajectory method for Lyapunov exponent estimation
    3. Statistical averaging for robust results
    4. Spectral analysis for complementary chaos detection
    5. Energy conservation validation
    
    The implementation provides reliable chaos detection for orbital
    trajectories around irregular gravitational bodies like asteroids,
    enabling assessment of long-term orbital stability and predictability.
    """)
