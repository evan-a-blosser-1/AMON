import numpy as np    

################################################
################################################
# page 7 paragraph 2: 2021 Aljibea & Sanchez First Approximations of Apophis 
# " We distributed our initial conditions in the y-axis, with x0 = z0 = y_dot_0 = z_dot_0 =
#  0 and x_dot_ 0 was computed according to Eq. 1." 
def v_calc(Ham,omega,mu_I,CM,yp):
    U = np.zeros(1, dtype="float64")
    for it in range(len(CM)):
        x = 0  - CM[it,0]
        y = yp - CM[it,1]
        z = 0  - CM[it,2]
        r = np.sqrt(x**2 + y**2 + z**2) 
        U += mu_I[it]/r
    #########################
    psu = U[0]
    centri = (omega**2)*(y**2)
    # print(f"Omega: {omega} rad/s")
    # print(f"Ham:      {Ham} (km^2/s^2) ")
    # print(f"Psuedo:   {psu} (km^2/s^2) ")
    # print(f"Centrifu: {centri} (km^2/s^2) ")
    arg = 2*Ham + centri + 2*psu
    if arg > 0:
        V = np.sqrt(arg)
    return V


def Calc_Ham(state,omega,mu_I,CM):
    U = np.zeros(1, dtype="float64")
    for it in range(len(CM)):
        x = state[0] - CM[it,0]
        y = state[1] - CM[it,1]
        z = state[2] - CM[it,2]
        r = np.sqrt(x**2 + y**2 + z**2) 
        U += mu_I[it]/r
    ############################
    X = state[0]
    Y = state[1]
    Z = state[2]
    VX = state[3]
    VY = state[4]
    VZ = state[5]
    V_mag = np.sqrt(VX**2 + VY**2 + VZ**2)
    Energy = 0.5*(VX**2 + VY**2 + VZ**2) - 0.5*omega**2* (X**2 + Y**2 + Z**2) - U[0]
    # print(Energy)
    return Energy