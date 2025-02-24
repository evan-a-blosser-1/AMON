import numpy as np    

################################################
################################################
# page 7 paragraph 2
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