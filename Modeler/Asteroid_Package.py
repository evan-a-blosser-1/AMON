""" Asteroid Package
    ---
    Contains Functions used for handling Alias Waveform 
    Format (.obj) files & functions for plotting orbits 
    -----------------------
    Author: Evan A. Blosser                              
    Date:   Fri Jan 5 2024                     
    -----------------------
     
    Function list: (assuming import as Astro)
    ---
    
    - OBJ file to Vertices & Faces

    
    ```python
    >>> Astro.OBJ_2_VertFace(Asteroid_file)
    
    ```
    
    - OBJ file to 3D Mesh (for poly-collection plotting)
    
    ```python
    >>> Astro.OBJ_2_Mesh(Asteroid_Name,scale,Mesh_color)
    ```
    
    
    - Orbital Gauge Cluster
    
    ```python
    >>> Astro.Orbit_Gauge_Cluster(Orbital_Elements, 
                                    a, mu,R_body, 
                                    t_inp,Graph_time,
                                    Graph_Time_Unit,
                                    Aster_File_Name)
    ```
                                    
    - Basic 3D Orbit Plot
    
    ```python
    >>> Astro.Orbit_3D_Plot(a)
    ```
        
    - MASCON 3D Orbit Plot
    
    ```python
    >>> Astro.MASCON_Orbit_3D(Initial_Conditions, 
                                MASCON_Choice, 
                                Center_of_mass, a, 
                                File_OBJ_Dir,scale)
    ```   
    
    - Triaxial Ellipsoid Gravity Potential Check   
    
    ```python
    >>> Astro.Triax_Ellip(Ax, x_in, Sphere_Override)
    ```   
"""
############################# Imports
#####################################
# Mathmatical!!                     #
import numpy as np                  #
from scipy.optimize import root     #
from scipy.special import elliprj   #
from scipy.special import elliprf   #
# System Time                       #
import time  
import sys

# Plotting & Animation              #####################
import matplotlib.pyplot as plt                         #
from mpl_toolkits.mplot3d.art3d import Poly3DCollection #
import matplotlib.ticker as ticker                      #
#########################################################

#%% Sim Progess Bar

def print_progress(current_time, total_time, bar_length=11):
    fraction = current_time / total_time
    block = int(round(bar_length * fraction))
    bar = '\u25A1' * block + '-' * (bar_length - block)
    sys.stdout.write(f'\r[{bar}] {fraction:.2%}')
    sys.stdout.flush()
           
# File Handling 
def Loading():
    # Define the frames of the animation
    frames = ['-', '\\', '|', '/']

    # Loop to create the animation
    while True:
        for frame in frames:
            # Print the frame and flush the output
            sys.stdout.write('\r' + frame)
            sys.stdout.flush()
            # Delay for a short period
            time.sleep(0.1)
            
            
#%% volInt.c File Make

def volInt_Input_FileMake(Asteroid_file_in, PATH_Name):
    """ Formerly obj2MirtichData.py
    
        generates input file for volInt.c to be called inside Python

    Args:
        Asteroid_file_in (File and Path): Input File 
        PATH_Name (Path): Save path for the .in file.
    """
    ###############################################################
    # Add .obj extension to file name                             
    Asteroid_file = (Asteroid_file_in)
    #######################
    # Load ,txt data file ###################################### 
    data = np.loadtxt(Asteroid_file, delimiter=' ', dtype=str) 
    ############################################################
    ################################
    # Determine where Vertices End # 
    ################################
    # This section allows for any .txt in the .OBJ format file to be loaded
    #  and used for analysis 
    ###############################################################
    # Set Vertex/Faces denotaed as v or f in .obj format to array #
    vertex_faces = data[:,0]                                      #
    # Get Length of the Vertex/Faces array for range counting     #
    V_F_Range = vertex_faces.size                                 #
    # Define varibale for number of vertices & faces              #
    numb_vert = 0                                                 #
    numb_face = 0                                                 #
    # Scan Data for v & f and count the numbers of each.          #
    #  Used for sorting x, y, & z as vertices                     #
    for i in range(0,V_F_Range):                                  #
        if vertex_faces[i] == 'v':                                #
            numb_vert += 1                                        #
        else:                                                     #
            numb_face += 1                                        #
    ###############################################################
    #
    #########################
    # Assigning Vertex Data #
    #########################
    # Vertex data assigned to x, y, & z
    #  then cpnverts to float type
    ########################################
    # Assign 2nd row of .txt as x input    #
    x_input = data[range(0,numb_vert),1]   #
    # Assign 3rd row of .txt as y input    #
    y_input = data[range(0,numb_vert),2]   #
    # Assign 4th row of .txt as z input    #
    z_input = data[range(0,numb_vert),3]   #
    # Convert Vertices data to float type  #
    x = x_input.astype(float)             #
    y = y_input.astype(float)            #
    z = z_input.astype(float)           #
    ####################################
    #######################
    # Assigning Face Data #
    #######################
    # Face data assigned to fx, fy, & fz
    #  then cpnverts to float type
    #############################################
    # Range count for face data                 #
    row_tot = numb_face + numb_vert             #
    # Assign 2nd row of .txt as x input         #
    fx_input = data[range(numb_vert,row_tot),1] #
    # Assign 3rd row of .txt as y input         #
    fy_input = data[range(numb_vert,row_tot),2] #
    # Assign 4th row of .txt as z input         #
    fz_input = data[range(numb_vert,row_tot),3] #
    # Convert Vertices data to float type       #
    fx = fx_input.astype(str)                   #
    fy = fy_input.astype(str)                  #
    fz = fz_input.astype(str)                 #
    # Define: number of vertices on ith face #
    #  - used for .C program                 # 
    ith_face = []                            #
    for j in range(numb_vert,row_tot):       #
        ith_face.append(3)                   #
    ith_array = np.array(ith_face)           #
    ##########################################
    ##########################
    # Creating Output Arrays #
    ##########################
    #    Number of Vertex is (N-1)             
    #     numb_vert += 1
    #########################################
    # Number of Vertex set to array         
    numb_vert_array = []                    
    numb_vert_array.append(numb_vert)       
    # Number of Faces set to array          
    numb_face_array = []                    
    numb_face_array.append(numb_face)       
    # Stacking Columns of Vertex Data       
    Vert_Data_Out_0 = np.column_stack((x, y))               
    Vert_Data_Out   = np.column_stack((Vert_Data_Out_0, z)) 
    # Stacking Columns of Face Data                         
    Face_Data_Out_0 = np.column_stack((ith_array, fx))      
    Face_Data_Out_Y = np.column_stack((Face_Data_Out_0,fy)) 
    Face_Data_Out   = np.column_stack((Face_Data_Out_Y,fz)) 
    #########################################################################
    ################################# Write Data File for Mirtich's Program 
    Save_Path = PATH_Name + ".in"                        
    with open(Save_Path,"w") as Poly_Data_file:                
        np.savetxt(Poly_Data_file,numb_vert_array,fmt='%s',delimiter='\t'); 
        np.savetxt(Poly_Data_file,Vert_Data_Out,fmt='%.8f',delimiter='\t'); 
        np.savetxt(Poly_Data_file,numb_face_array,fmt='%s',delimiter=' ');  
        np.savetxt(Poly_Data_file,Face_Data_Out,fmt='%s',delimiter=' ');    
    #########################################################################
 

#%% OBJ file to Vertices & Faces 
###################################################################################
#################################### OBJ to Vertices & Faces ######################
def OBJ_2_VertFace(Asteroid_file):
    """Reads an OBJ file in and 
        converts the vertice and
        face data into arrays.
    Args:
        Asteroid_file (file): .obj file
    Returns:
        numpy array: 2 arrays of Vertice and Face data
    """
    ##############################################################
    OBJ_Data = np.loadtxt(Asteroid_file, delimiter=' ', dtype=str) 
    ######################
    ## Process OBJ File ##
    ###############################################################
    # Set Vertex/Faces denotaed as v or f in .obj format to array #
    vertex_faces = OBJ_Data[:,0]                                  #
    # Get Length of the Vertex/Faces array for range counting     #
    V_F_Range = vertex_faces.size                                 #
    # Define varibale for number of vertices & faces              #
    numb_vert = 0                                                 #
    numb_face = 0                                                 #
    # Scan Data for v & f and count the numbers of each.          #
    #  Used for sorting x, y, & z as vertices                     #
    for i in range(0,V_F_Range):                                  #
        if vertex_faces[i] == 'v':                                #
            numb_vert += 1                                        #
        else:                                                     #
            numb_face += 1                                        #
    ###############################################################
    #########################
    # Assigning Vertex Data #
    #########################
    # Vertex data assigned to x, y, & z
    #  then cpnverts to float type
    ########################################
    # Assign 2nd row of .txt as x input    
    x_input = OBJ_Data[range(0,numb_vert),1]   
    # Assign 3rd row of .txt as y input    
    y_input = OBJ_Data[range(0,numb_vert),2]   
    # Assign 4th row of .txt as z input    
    z_input = OBJ_Data[range(0,numb_vert),3] 
    ########################################  
    # Convert Vertices data to float type  #
    x_0 = x_input.astype(float)            #
    y_0 = y_input.astype(float)            #
    z_0 = z_input.astype(float)            #
    ########################################
    #
    ##############################################
    # Fill zero indecies with dummy values       #
    #  to allow faces to call vertices 1 to 1014 #
    x = np.append(0,x_0)  ########################
    y = np.append(0,y_0)  #
    z = np.append(0,z_0)  #
    #######################
    #
    #######################
    # Assigning Face Data #
    #######################
    # Face data assigned to fx, fy, & fz
    #  then cpnverts to float type
    #############################################
    # Range count for face data                 
    row_tot = numb_face + numb_vert             
    # Assign 2nd row of .txt as x input         
    fx_input = OBJ_Data[range(numb_vert,row_tot),1] 
    # Assign 3rd row of .txt as y input         
    fy_input = OBJ_Data[range(numb_vert,row_tot),2] 
    # Assign 4th row of .txt as z input         
    fz_input = OBJ_Data[range(numb_vert,row_tot),3] 
    #######################################
    # Convert Vertices data to float type #
    fx = fx_input.astype(int)             #
    fy = fy_input.astype(int)             #
    fz = fz_input.astype(int)             #
    #######################################
    #
    ##########################
    # Creating Output Arrays #
    ##########################
    #    Number of Vertex is (N-1)             
    #     numb_vert += 1
    #########################################
    # Number of Vertex set to array         #
    numb_vert_array = []                    #
    numb_vert_array.append(numb_vert)       #
    # Number of Faces set to array          #
    numb_face_array = []                    #
    numb_face_array.append(numb_face)       #
    # Stacking Columns of Vertex Data       #################
    Vert_Data_Out_0 = np.column_stack((x, y))               #
    Vert_Data       = np.column_stack((Vert_Data_Out_0, z)) #
    # Stacking Columns of Face Data                         #
    Face_Data_Out_0 = np.column_stack((fx,fy))              #
    Face_Data       = np.column_stack((Face_Data_Out_0,fz)) #
    #########################################################
    return Vert_Data , Face_Data

#%% OBJ file to 3D Mesh (for poly-collection plotting)
#################################################################################
################################################## OBJ Mesh Read In #############
def OBJ_2_Mesh(Asteroid_Name,scale,Mesh_color):
    """ This reads in .obj files and creates the face mesh
     That is used for plotting around the Center of Masses
     
    Args:
        Asteroid_Name (.obj file):  Asteroid Shape model File
        
        scale               flaot:  Scale for asteroid in 
                                    kilometers. 1 if no scaling.
                                    
        color              string:  Plot color for edges of mesh
    Returns:
        Mesh: Asteroid face mesh 
    """
    #######################
    # Load data file      ###################################### 
    data = np.loadtxt(Asteroid_Name, delimiter=' ', dtype=str) #
    ############################################################
    ###############################################################
    # Set Vertex/Faces denotaed as v or f in .obj format to array #
    vertex_faces = data[:,0]                                      #
    # Get Length of the Vertex/Faces array for range counting     #
    V_F_Range = vertex_faces.size                                 #
    # Define varibale for number of vertices & faces              #
    numb_vert = 0                                                 #
    numb_face = 0                                                 #
    # Scan Data for v & f and count the numbers of each.          #
    #  Used for sorting x, y, & z as vertices                     #
    for i in range(0,V_F_Range):                                  #
        if vertex_faces[i] == 'v':                                #
            numb_vert += 1                                        #
        else:                                                     #
            numb_face += 1                                        #
    ###############################################################
    #########################
    # Assigning Vertex Data #
    #########################
    # Vertex data assigned to x, y, & z
    #  then cpnverts to float type
    ########################################
    # Assign 2nd row of .txt as x input    #
    x_input = data[range(0,numb_vert),1]   #
    # Assign 3rd row of .txt as y input    #
    y_input = data[range(0,numb_vert),2]   #
    # Assign 4th row of .txt as z input    #
    z_input = data[range(0,numb_vert),3]   #
    # Convert Vertices data to float type  #
    x_0 = x_input.astype(float)            #
    y_0 = y_input.astype(float)            #
    z_0 = z_input.astype(float)            #
    ########################################
    #
    ##############################################
    # Fill zero indecies with dummy values       #
    #  to allow faces to call vertices 1 to 1014 #
    x = np.append(0,x_0)  ########################
    y = np.append(0,y_0)  #
    z = np.append(0,z_0)  #
    #######################
    #
    #######################
    # Assigning Face Data #
    #######################
    # Face data assigned to fx, fy, & fz
    #  then cpnverts to float type
    #############################################
    # Range count for face data                 #
    row_tot = numb_face + numb_vert             #
    # Assign 2nd row of .txt as x input         #
    fx_input = data[range(numb_vert,row_tot),1] #
    # Assign 3rd row of .txt as y input         #
    fy_input = data[range(numb_vert,row_tot),2] #
    # Assign 4th row of .txt as z input         #
    fz_input = data[range(numb_vert,row_tot),3] #
    # Convert Vertices data to float type       #
    fx = fx_input.astype(int)                   #
    fy = fy_input.astype(int)                  #
    fz = fz_input.astype(int)                 #
    ##########################################
    #
    ##########################
    # Creating Output Arrays #
    ##########################
    #    Number of Vertex is (N-1)             
    #     numb_vert += 1
    #########################################
    # Number of Vertex set to array         #
    numb_vert_array = []                    #
    numb_vert_array.append(numb_vert)       #
    # Number of Faces set to array          #
    numb_face_array = []                    #
    numb_face_array.append(numb_face)       #
    # Stacking Columns of Vertex Data       #################
    Vert_Data_Out_0 = np.column_stack((x, y))               #
    Vert_Data       = np.column_stack((Vert_Data_Out_0, z)) #
    # Stacking Columns of Face Data                         #
    Face_Data_Out_0 = np.column_stack((fx,fy))              #
    Face_Data       = np.column_stack((Face_Data_Out_0,fz)) #
    #########################################################
    Vert_Data_mesh = Vert_Data*scale
    ##:)                                                 
    # Let's put a Happy little Asteroid right in there        
    Asteroid_Mesh = Poly3DCollection([Vert_Data_mesh[ii] for ii in Face_Data], 
                            edgecolor=Mesh_color,
                            facecolors="white",
                            linewidth=0.75,
                            alpha=0.0)
    ######################################
    return Asteroid_Mesh



#%% Orbital Gauge Cluster
#################################################################################
############################################### Astro Gauge Closter #############
def Orbit_Gauge_Cluster(Orbital_Elements, a, mu,R_body, t_inp,Graph_time,Graph_Time_Unit,Aster_File_Name):
  """Orbital Gauge CLuster
      This is a sub-plot of orbital elements along with plot limits.

  Args:
      Orbnital_Elements (Pandas DataFrame): A DataFrame of Orbital Elements to be plotted. 
      a (array): ODEINT output in the form of the state vector.
      mu (flaot): Gravitational Parameter of body being orbited.
      R_body (flaot): Radius of body being orbited.
      t_inp (flaot): Duration of plot.
      Graph_time (flaot): Conversion to seconds selected.
      Graph_Time_Unit (string): The unit of time for x-axis label.
      Aster_File_Name (string): The name of the asteroid being plotted.
  Returns:
        Sub-Plots: Orbital-Gauge-Cluster! Don't forget to call the plot with `plt.show()` !!
  """
  ##########
  # Colors #
  ##########
  # COE Output Plots
  Tick_Mark_Col    = '#1A85FF'
  major_grid_col   = "#00E2F9"
  minor_grid_col   = "#00C8DC"
  Background       = "#000000"
  Font_color       = "#02FF1F"
  Plot_line_col    = "#f2ffb7"
  Limit_Line_Color = '#ff06b5'
  ###########################
  # Fromat Graph_Time 
  # Fromat Graph_Time 
  # n = t_inp*Graph_time
  t_span  = np.linspace(0,t_inp,t_inp)
  # Formated_Graph_Time = t_span/Graph_time
  Formated_Graph_Time = t_span
  #############################
  # Orbital Elements Plotting #
  ################################################################
  # Set subplot                                                  
  figure1, axis = plt.subplots( 3, 3,  figsize=(12, 10),         
                        facecolor=Background)   
  # Figure MAIN Title
  figure1.suptitle(f'Asteroid: {Aster_File_Name} Orbit Simulated: {t_inp} {Graph_Time_Unit}',
                  fontsize=16, fontweight='bold',fontdict={'family': 'Consolas'},color=Font_color)                
  # Window Title                                                 
  figure1.canvas.manager.set_window_title(                       
      'Orbital Gauge Cluster')                                 
  #####################
  # Angular Momentum  ################
  axis[0, 0].plot(Formated_Graph_Time,Orbital_Elements[0], color=Plot_line_col) 
  # Tick Settings  
  axis[0, 0].xaxis.set_minor_locator(ticker.AutoMinorLocator())
  axis[0, 0].yaxis.set_minor_locator(ticker.AutoMinorLocator())
  # Labels & Colors         
  axis[0, 0].set_title("Angular Momentum (h)",
                      fontsize=14, fontweight='bold',fontdict={'family': 'Consolas'},color=Font_color)                     
  axis[0, 0].set_xlabel('Time ({})'.format(Graph_Time_Unit) ,
                        fontsize=12, fontweight='light',fontdict={'family': 'Consolas'},color=Font_color)                                  
  axis[0, 0].set_ylabel('\u0394h (km^2/s)',
                        fontsize=12, fontweight='light',fontdict={'family': 'Consolas'},color=Font_color) 
  axis[0, 0].tick_params(axis='x', colors=Tick_Mark_Col) 
  axis[0, 0].tick_params(axis='y', colors=Tick_Mark_Col)  
  axis[0, 0].grid(which='major', linestyle='--',linewidth=0.5, color=major_grid_col)
  axis[0, 0].grid(which='minor', linestyle=':', linewidth='0.5', color=minor_grid_col)
  axis[0, 0].set_facecolor(Background) 
  ###############################
  #               
  ###########################
  # Semi-Major Axis Plot    ###############
  axis[0, 1].plot(Formated_Graph_Time,Orbital_Elements[1], color=Plot_line_col)
  # Tick Settings  
  axis[0, 1].xaxis.set_minor_locator(ticker.AutoMinorLocator())
  axis[0, 1].yaxis.set_minor_locator(ticker.AutoMinorLocator())    
  # Labels & Colors            
  axis[0, 1].set_title("Semi-Major Axis (A)",
                      fontsize=14, fontweight='bold',fontdict={'family': 'Consolas'},color=Font_color)                     
  axis[0, 1].set_xlabel('Time ({})'.format(Graph_Time_Unit)  ,
                        fontsize=12, fontweight='light',fontdict={'family': 'Consolas'},color=Font_color)                                  
  axis[0, 1].set_ylabel('\u0394A (km)',
                        fontsize=12, fontweight='light',fontdict={'family': 'Consolas'},color=Font_color) 
  axis[0, 1].tick_params(axis='x', colors=Tick_Mark_Col) 
  axis[0, 1].tick_params(axis='y', colors=Tick_Mark_Col)  
  axis[0, 1].grid(which='major', linestyle='--',linewidth=0.5, color=major_grid_col)
  axis[0, 1].grid(which='minor', linestyle=':', linewidth='0.5', color=minor_grid_col)                  
  axis[0, 1].set_facecolor(Background)  
  #################################
  #
  ######################
  # Eccentricity Plot  ################
  #####################################
  # Set limit at e = 0.05
  Eccentricity_Limit = 0.05* np.ones_like(Formated_Graph_Time+30)
  # Plot Limit
  axis[0, 2].plot(Formated_Graph_Time, Eccentricity_Limit,linestyle='dashdot',
                  color=Limit_Line_Color,label='Ecc. Lim')
  # Set Legend
  Leg_Alt = axis[0, 2].legend(loc='upper right', bbox_to_anchor=(1.2, 1.15))
  Leg_Alt.get_frame().set_facecolor(Background)  
  Leg_Alt.get_texts()[0].set_color(Font_color)  
  # Plot Data
  axis[0, 2].plot(Formated_Graph_Time,Orbital_Elements[2], color=Plot_line_col) 
  # Tick Settings  
  axis[0, 2].xaxis.set_minor_locator(ticker.AutoMinorLocator())
  axis[0, 2].yaxis.set_minor_locator(ticker.AutoMinorLocator())
  # Labels & Colors             
  axis[0, 2].set_title("Eccentricity:",
                      fontsize=14, fontweight='bold',fontdict={'family': 'Consolas'},color=Font_color)                     
  axis[0, 2].set_xlabel('Time ({})'.format(Graph_Time_Unit)  ,
                        fontsize=12, fontweight='light',fontdict={'family': 'Consolas'},color=Font_color)                                  
  axis[0, 2].set_ylabel('\u0394e',
                        fontsize=12, fontweight='light',fontdict={'family': 'Consolas'},color=Font_color) 
  axis[0, 2].tick_params(axis='x', colors=Tick_Mark_Col) 
  axis[0, 2].tick_params(axis='y', colors=Tick_Mark_Col)  
  axis[0, 2].grid(which='major', linestyle='--',linewidth=0.5, color=major_grid_col)
  axis[0, 2].grid(which='minor', linestyle=':', linewidth='0.5', color=minor_grid_col)                
  axis[0, 2].set_facecolor(Background)     
  ###############################
  #
  ######################
  # Inclination Plot   ######################### 
  ###############################################
  # Set limit at i = 0.001
  Inclination_Limit = 0.001 * np.ones_like(Formated_Graph_Time)
  # Plot Limit
  axis[1, 0].plot(Formated_Graph_Time, Inclination_Limit,linestyle='dashdot',
                  color=Limit_Line_Color,label='Inc. lim')
  # Set Legend
  Leg_Alt = axis[1, 0].legend(loc='upper right', bbox_to_anchor=(1.2, 1.15))
  Leg_Alt.get_frame().set_facecolor(Background)  
  Leg_Alt.get_texts()[0].set_color(Font_color)  
  # Plot Data                                  
  axis[1, 0].plot(Formated_Graph_Time,Orbital_Elements[3], color=Plot_line_col)     
  # Tick Settings  
  axis[1, 0].xaxis.set_minor_locator(ticker.AutoMinorLocator())
  axis[1, 0].yaxis.set_minor_locator(ticker.AutoMinorLocator())
  # Labels & Colors       
  axis[1, 0].set_title("Inclination:",
                      fontsize=14, fontweight='bold',fontdict={'family': 'Consolas'},color=Font_color)                     
  axis[1, 0].set_xlabel('Time ({})'.format(Graph_Time_Unit)  ,
                        fontsize=12, fontweight='light',fontdict={'family': 'Consolas'},color=Font_color)                                  
  axis[1, 0].set_ylabel('\u0394i (degrees)',
                        fontsize=12, fontweight='light',fontdict={'family': 'Consolas'},color=Font_color) 
  axis[1, 0].tick_params(axis='x', colors=Tick_Mark_Col) 
  axis[1, 0].tick_params(axis='y', colors=Tick_Mark_Col)  
  axis[1, 0].grid(which='major', linestyle='--',linewidth=0.5, color=major_grid_col)
  axis[1, 0].grid(which='minor', linestyle=':', linewidth='0.5', color=minor_grid_col)                  
  axis[1, 0].set_facecolor(Background)     
  ##################################
  #
  ####################################
  # Longitude of Ascending Node Plot ##############
  axis[1, 1].plot(Formated_Graph_Time,Orbital_Elements[4], color=Plot_line_col)
  # Tick Settings  
  axis[1, 1].xaxis.set_minor_locator(ticker.AutoMinorLocator())
  axis[1, 1].yaxis.set_minor_locator(ticker.AutoMinorLocator())  
  # Labels & Colors       
  axis[1, 1].set_title("Longitude of Ascending Node (\u03A9)",
                      fontsize=14, fontweight='bold',fontdict={'family': 'Consolas'},color=Font_color)                     
  axis[1, 1].set_xlabel('Time ({})'.format(Graph_Time_Unit)  ,
                        fontsize=12, fontweight='light',fontdict={'family': 'Consolas'},color=Font_color)                                  
  axis[1, 1].set_ylabel('\u0394\u03A9 (degrees)',
                        fontsize=12, fontweight='light',fontdict={'family': 'Consolas'},color=Font_color) 
  axis[1, 1].tick_params(axis='x', colors=Tick_Mark_Col) 
  axis[1, 1].tick_params(axis='y', colors=Tick_Mark_Col)  
  axis[1, 1].grid(which='major', linestyle='--',linewidth=0.5, color=major_grid_col)
  axis[1, 1].grid(which='minor', linestyle=':', linewidth='0.5', color=minor_grid_col)
  axis[1, 1].set_facecolor(Background)  
  ##################################
  #  
  ########################
  # Orbital Details List #
  #############################################################
  # Period Calc ###############################################
  ax_avg            = np.mean(Orbital_Elements[1])
  ax_uncert         = (np.std(Orbital_Elements[1])/ax_avg)*100
  ORB_PER_ARG       = (ax_avg**3)/mu
  if ORB_PER_ARG < 0:
      print(ORB_PER_ARG)
      ax_avg = -ax_avg
  Orbit_Period      = 2*np.pi*np.sqrt((ax_avg**3)/mu)
  Total_Sim_Time    = t_inp*Graph_time
  Number_Of_Orbits  = Total_Sim_Time/Orbit_Period
  Orbit_Period_unit = 'Sec.'
  # Change Units of Period
  if Orbit_Period > 120:
      Orbit_Period_min = Orbit_Period/60
      Orbit_Period     = Orbit_Period_min
      Orbit_Period_unit = 'Min.'
      if Orbit_Period_min > 120:
          Orbit_Period_hr = Orbit_Period_min/60
          Orbit_Period     = Orbit_Period_hr
          Orbit_Period_unit = 'Hrs.'
          if Orbit_Period_hr > 48:
              Orbit_Period_day = Orbit_Period_hr/24
              Orbit_Period     = Orbit_Period_day
              Orbit_Period_unit = 'Days' 
  ############################################################
  # Orbit Energy #############################################
  Orb_ecc       = np.mean(Orbital_Elements[2])
  ecc_uncert    = (np.std(Orbital_Elements[2])/Orb_ecc)*100
  Energy_uncert = ax_uncert + ecc_uncert
  Orb_apogee    = ax_avg*(1 + Orb_ecc)
  Orb_perigee   = ax_avg*(1 - Orb_ecc)
  Orbit_Energy  = - mu/(Orb_apogee + Orb_perigee)
  ############################################################
  # Pro/Retro    #############################################
  inclin_avg = np.mean(Orbital_Elements[3])
  Orbit_Type_Output ="N/A"
  if inclin_avg < 90:
      Orbit_Type_Output = 'PROGRADE'  
  elif inclin_avg > 90:
      Orbit_Type_Output = 'RETROGRADE'  
  elif inclin_avg == 90:
      Orbit_Type_Output = 'POLAR'
  #######################################################################
  ########################## Text Output ################################
  Display_Text_Out = [f"""
  {'-'*42}
  {Orbit_Type_Output} orbit 
  {'-'*42}
  {'-'*42}
  Period = {Orbit_Period:.3f} {Orbit_Period_unit} 
  (+/-) {ax_uncert:.2f} %
  {'-'*42}
  Number of simulated orbits: {Number_Of_Orbits:.2f} 
  {'-'*42}
  Energy = {Orbit_Energy:.3e} km^2/s^2 
  (+/-) {Energy_uncert:.2f} % 
  {'-'*42}
  """
  ]
  axis[1, 2].text(0.5, 0.5, Display_Text_Out[0],
                              horizontalalignment='center',
                              verticalalignment='center',
                              fontsize=12, color= Font_color)
  axis[1, 2].set_facecolor(Background)
  ##################################################################
  ##################################################################

  #############################
  # Argument of Perigee Plot  ###############
  axis[2, 0].plot(Formated_Graph_Time,Orbital_Elements[5], color=Plot_line_col)    
  # Tick Settings  
  axis[2, 0].xaxis.set_minor_locator(ticker.AutoMinorLocator())
  axis[2, 0].yaxis.set_minor_locator(ticker.AutoMinorLocator())    
  # Labels & Colors        
  axis[2, 0].set_title("Argument of Perigee (\u03C9)",
                      fontsize=14, fontweight='bold',fontdict={'family': 'Consolas'},color=Font_color)                     
  axis[2, 0].set_xlabel('Time ({})'.format(Graph_Time_Unit)  ,
                        fontsize=12, fontweight='light',fontdict={'family': 'Consolas'},color=Font_color)                                  
  axis[2, 0].set_ylabel('\u0394\u03C9 (degrees)',
                        fontsize=12, fontweight='light',fontdict={'family': 'Consolas'},color=Font_color) 
  axis[2, 0].tick_params(axis='x', colors=Tick_Mark_Col) 
  axis[2, 0].tick_params(axis='y', colors=Tick_Mark_Col)  
  axis[2, 0].grid(which='major', linestyle='--',linewidth=0.5, color=major_grid_col)
  axis[2, 0].grid(which='minor', linestyle=':', linewidth='0.5', color=minor_grid_col)      
  axis[2, 0].set_facecolor(Background) 
  #################################
  #
  #####################
  # True Anomaly Plot ##########
  axis[2, 1].plot(Formated_Graph_Time,Orbital_Elements[6], color=Plot_line_col)     
  # Tick Settings  
  axis[2, 1].xaxis.set_minor_locator(ticker.AutoMinorLocator())
  axis[2, 1].yaxis.set_minor_locator(ticker.AutoMinorLocator())       
  # Labels & Colors            
  axis[2, 1].set_title("True Anomaly (\u03BD)",
                      fontsize=14, fontweight='bold',fontdict={'family': 'Consolas'},color=Font_color)                     
  axis[2, 1].set_xlabel('Time ({})'.format(Graph_Time_Unit)  ,fontsize=12, 
                        fontweight='light',fontdict={'family': 'Consolas'},color=Font_color)                                  
  axis[2, 1].set_ylabel('\u0394\u03BD (degrees)',
                        fontsize=12, fontweight='light',fontdict={'family': 'Consolas'},color=Font_color) 
  axis[2, 1].tick_params(axis='x', colors=Tick_Mark_Col) 
  axis[2, 1].tick_params(axis='y', colors=Tick_Mark_Col)  
  axis[2, 1].grid(which='major', linestyle='--',linewidth=0.5, color=major_grid_col)
  axis[2, 1].grid(which='minor', linestyle=':', linewidth='0.5', color=minor_grid_col)
  axis[2, 1].set_facecolor(Background) 
  ####################################
  #
  ##################################
  # Altimeter!! Big Brain Idea lol ###########################
  ############################################################
  # Set limit at mean radius of asteroid, make a list of 
  # these points to graph
  Altimeter_Limit = R_body* np.ones_like(Formated_Graph_Time)
  # Plot Limit
  axis[2, 2].plot(Formated_Graph_Time, Altimeter_Limit,linestyle='dashdot',
                  color=Limit_Line_Color,label='Crash Lim')
  # Set Legend
  Leg_Alt = axis[2, 2].legend(loc='upper right', bbox_to_anchor=(1.2, 1.15))
  Leg_Alt.get_frame().set_facecolor(Background)  
  Leg_Alt.get_texts()[0].set_color(Font_color)   
  # Find Altitude:
  Altitude_Sim = (a[:,0]**2 + a[:,1]**2 + a[:,2]**2 )**(1/2)
  # Plot Data
  axis[2, 2].plot(Formated_Graph_Time,Altitude_Sim, color=Plot_line_col)     
  # Tick Settings  
  axis[2, 2].xaxis.set_minor_locator(ticker.AutoMinorLocator())
  axis[2, 2].yaxis.set_minor_locator(ticker.AutoMinorLocator())        
  # Labels & Colors            
  axis[2, 2].set_title("Altimeter:",
                      fontsize=14, fontweight='bold',fontdict={'family': 'Consolas'},color=Font_color)                     
  axis[2, 2].set_xlabel('Time ({})'.format(Graph_Time_Unit)  ,fontsize=12, 
                        fontweight='light',fontdict={'family': 'Consolas'},color=Font_color)                                  
  axis[2, 2].set_ylabel('Altitude (km)',
                        fontsize=12, fontweight='light',fontdict={'family': 'Consolas'},color=Font_color) 
  axis[2, 2].tick_params(axis='x', colors=Tick_Mark_Col) 
  axis[2, 2].tick_params(axis='y', colors=Tick_Mark_Col)  
  axis[2, 2].grid(which='major', linestyle='--',linewidth=0.5, color=major_grid_col)
  axis[2, 2].grid(which='minor', linestyle=':', linewidth='0.5', color=minor_grid_col)
  axis[2, 2].set_facecolor(Background) 

  ###########################################################
  #################### Combine all the operations and display                       
  plt.tight_layout()    
  plt.show()                             
  ################################################################
#%% Basic 3D Orbit Plot
def Orbit_3D_Plot(a):
  """MASCON Orbital plot in 3-Dimensions
      Used to plot simulated orbit around the tetrahedron center of masses,
      contianed within the asteroid's outer mesh.
  Args:
      a (array): State Vector of Orbit.
  Returns:
      plot: 3D plot of the orbit aroudn teh asteorid shape model. Don't forget to call the plot with `plt.show()` !! 
  """
  ############
  # Settings #
  #####################
  # Colors
  grid_col   = '#0200FF'
  Space      = "#000000"
  orbit_line = "#F70101"
  # Grid Color                            
  plt.rcParams['grid.color'] = grid_col   
  #####################################
  # Set plot                              
  fig = plt.figure('Orbit')               
  # Set axis                              
  axis = plt.axes(projection='3d')           
  # Set Window Size                       
  fig.tight_layout()              
  # Plot assumed center                   
  cm_x = 0
  cm_y = 0
  cm_z = 0
  axis.scatter3D(cm_x,cm_y,cm_z,
                  marker='o',
                  color='#D41159')
  ##########################################
  ############################## Set Aspect
  axis.set_box_aspect([1,1,1])
  #########################################
  # Set x data to position, i. given by a #
  xline = a[:,0]                          #
  # Set y data to position, j. given by a #
  yline = a[:,1]                          #
  # Set z data to position, k. given by a #
  zline = a[:,2]                          #
  # Plot line and asteroid                #
  axis.plot3D(xline, yline, zline,        #
          color=orbit_line )              #
  # Axis Labels                           #
  axis.set_xlabel('x (km)')                 #
  axis.set_ylabel('y (km)')                  #
  axis.set_zlabel('z (km)')                   #
  axis.tick_params(axis='x', colors=grid_col) #
  axis.tick_params(axis='y', colors=grid_col) #
  axis.tick_params(axis='z', colors=grid_col) #
  axis.yaxis.label.set_color(grid_col)        #  
  axis.xaxis.label.set_color(grid_col)       #  
  axis.zaxis.label.set_color(grid_col)      #
  # Background Color                      # 
  fig.set_facecolor(Space)                #
  axis.set_facecolor(Space)                #
  # Grid Pane Color/set to clear          #
  axis.xaxis.set_pane_color((0.0, 0.0,     #
                              0.0, 0.0))  #
  axis.yaxis.set_pane_color((0.0, 0.0,     #
                              0.0, 0.0))  #
  axis.zaxis.set_pane_color((0.0, 0.0,     #
                              0.0, 0.0))  # 
  #########################################
  return
############################################

#%% MASCON 3D Orbit Plot
def MASCON_Orbit_3D(Initial_Conditions, MASCON_Choice, Center_of_mass, a, File_OBJ_Dir,scale):
  """MASCON Orbital plot in 3-Dimensions
        Used to plot simulated orbit around the tetrahedron center of masses,
        contianed within the asteroid's outer mesh.
  Args:
      Initial_Conditions (array): Tetrahedron center of masses.
      MASCON_Choice (int): This is a selection integer 1, 3, & 8 ONLY!!
      Center_of_mass (array): Polyhedron center of mass.
      a (array): State Vector of Orbit.
      File_OBJ_Dir (path & file name): The path & name of the .obj file, for the mesh.
      scale (float): mesh scale for the asteroid.

  Returns:
      plot: 3D plot of the orbit around the asteorid shape model. Don't forget to call the plot with `plt.show()` !!
  """
  ############
  # Settings #
  ############
  # CM point size in %
  Cm_plot_p_size   = 25
  # CM marker
  Cm_plot_mark_typ = '.'
  #####################
  # Colors
  grid_col   = '#0200FF'
  Space      = "#000000"
  orbit_line = "#F70101"
  Mesh_Color = '#FFB000'
  #############################
  Mascon_Color_Bank = ['#9600FF',
                      '#2600FF',
                      '#00D9FF',
                      '#00FF43',
                      '#D9FF00',
                      '#FFD500',
                      '#FFA700',
                      '#FFBB7D']
  ####################
  # 3D Plot of Orbit ######################
  # Grid Color                            #
  plt.rcParams['grid.color'] = grid_col   #
  #########################################
  # Set plot                              
  fig = plt.figure('Orbit')               
  # Set axis                              
  axis = plt.axes(121,projection='3d')              
  # Set Window Size                       
  fig.tight_layout()              
  # Plot CM of Asteroid                   
  cm_x = Center_of_mass[0]
  cm_y = Center_of_mass[1]
  cm_z = Center_of_mass[2]
  axis.scatter3D(cm_x,cm_y,cm_z,
                  marker='o',
                  color='#D41159')
  ############################
  # CM MASCON I, III, & VIII #
  ##########################################
  ################ MASCON I ################
  if MASCON_Choice == '1':
      axis.scatter3D(Initial_Conditions[:,0],
                  Initial_Conditions[:,1],
                  Initial_Conditions[:,2],
                  marker='o',
                  alpha=1,
                  s=Cm_plot_p_size, 
                  edgecolor=Mascon_Color_Bank[2])
  ############################################
  ################ MASCON III ################
  elif MASCON_Choice == '3':
      ###########
      # M1
      axis.scatter3D(Initial_Conditions[:,0],
                  Initial_Conditions[:,1],
                  Initial_Conditions[:,2],
                  marker=Cm_plot_mark_typ,
                  s=Cm_plot_p_size,
                  color=Mascon_Color_Bank[3])
      ##########
      # M2
      axis.scatter3D(Initial_Conditions[:,3],
                  Initial_Conditions[:,4],
                  Initial_Conditions[:,5],
              marker=Cm_plot_mark_typ,
                  s=Cm_plot_p_size,
                  color=Mascon_Color_Bank[4])
      ##########
      # M3
      axis.scatter3D(Initial_Conditions[:,6],
                  Initial_Conditions[:,7],
                  Initial_Conditions[:,8],
                  marker=Cm_plot_mark_typ,
                  s=Cm_plot_p_size,
                  color=Mascon_Color_Bank[5])
  #############################################
  ################ MASCON VIII ################
  else:
      ###########
      # M1
      axis.scatter3D(Initial_Conditions[:,0],
                  Initial_Conditions[:,1],
                  Initial_Conditions[:,2],
                  marker=Cm_plot_mark_typ,
                  s=Cm_plot_p_size,
                  color=Mascon_Color_Bank[0])
      ##########
      # M2
      axis.scatter3D(Initial_Conditions[:,3],
                  Initial_Conditions[:,4],
                  Initial_Conditions[:,5],
                  marker=Cm_plot_mark_typ,
                  s=Cm_plot_p_size,
                  color=Mascon_Color_Bank[1])
      ##########
      # M3
      axis.scatter3D(Initial_Conditions[:,6],
                  Initial_Conditions[:,7],
                  Initial_Conditions[:,8],
                  marker=Cm_plot_mark_typ,
                  s=Cm_plot_p_size,
                  color=Mascon_Color_Bank[2])
      ##########
      # M4
      axis.scatter3D(Initial_Conditions[:,9],
                  Initial_Conditions[:,10],
                  Initial_Conditions[:,11],
                  marker=Cm_plot_mark_typ,
                  s=Cm_plot_p_size,
                  color=Mascon_Color_Bank[3])
      ##########
      # M5
      axis.scatter3D(Initial_Conditions[:,12],
                  Initial_Conditions[:,13],
                  Initial_Conditions[:,14],
                  marker=Cm_plot_mark_typ,
                  s=Cm_plot_p_size,
                  color=Mascon_Color_Bank[4])
      ##########
      # M6
      axis.scatter3D(Initial_Conditions[:,15],
                  Initial_Conditions[:,16],
                  Initial_Conditions[:,17],
                  marker=Cm_plot_mark_typ,
                  s=Cm_plot_p_size,
                  color=Mascon_Color_Bank[5])
      ##########
      # M7
      axis.scatter3D(Initial_Conditions[:,18],
                  Initial_Conditions[:,19],
                  Initial_Conditions[:,20],
                  marker=Cm_plot_mark_typ,
                  s=Cm_plot_p_size,
                  color=Mascon_Color_Bank[6])
      ##########
      # M8
      axis.scatter3D(Initial_Conditions[:,21],
                  Initial_Conditions[:,22],
                  Initial_Conditions[:,23],
                  marker=Cm_plot_mark_typ,
                  s=Cm_plot_p_size,
                  color=Mascon_Color_Bank[7])
  #########################################
  # Set Aspect
  axis.set_box_aspect([1,1,1])
  # Add Asteroid MEsh
  # Read in OBJ file and assign as a 3D mesh of faces
  Asteroid_Mesh =OBJ_2_Mesh(
                            File_OBJ_Dir,
                            scale,
                            Mesh_Color)
  axis.add_collection3d(Asteroid_Mesh)
  #########################################
  fig2 = plt.figure('Orbit')               
  # Set axis      
  ax2 = plt.axes(122,projection='3d')   
  # Set x data to position, i. given by a #
  xline = a[:,0]                          #
  # Set y data to position, j. given by a #
  yline = a[:,1]                          #
  # Set z data to position, k. given by a #
  zline = a[:,2]                          #
  # Plot line and asteroid                #
  ax2.plot3D(xline, yline, zline,         #
          color=orbit_line )              #
  ax2.scatter3D(cm_x,cm_y,cm_z,
                  marker='o',
                  color='#D41159')
  ########################################
  ############################ Axis Labels
  axis.set_xlabel('x (km)')                 
  axis.set_ylabel('y (km)')                  
  axis.set_zlabel('z (km)')                            
  ax2.set_xlabel('x (km)')                 
  ax2.set_ylabel('y (km)')                  
  ax2.set_zlabel('z (km)') 
  ########################################
  ############################ Colors        
  axis.tick_params(axis='x', colors=grid_col) 
  axis.tick_params(axis='y', colors=grid_col) 
  axis.tick_params(axis='z', colors=grid_col) 
  axis.yaxis.label.set_color(grid_col)          
  axis.xaxis.label.set_color(grid_col)        
  axis.zaxis.label.set_color(grid_col)           
  ax2.tick_params(axis='x', colors=grid_col) 
  ax2.tick_params(axis='y', colors=grid_col) 
  ax2.tick_params(axis='z', colors=grid_col) 
  ax2.yaxis.label.set_color(grid_col)          
  ax2.xaxis.label.set_color(grid_col)        
  ax2.zaxis.label.set_color(grid_col)      
  # Background Color                      
  fig.set_facecolor(Space)                
  fig2.set_facecolor(Space) 
  axis.set_facecolor(Space) 
  ax2.set_facecolor(Space)                
  # Grid Pane Color/set to clear
  axis.xaxis.set_pane_color((0.0, 0.0,0.0, 0.0))  
  axis.yaxis.set_pane_color((0.0, 0.0,0.0, 0.0))  
  axis.zaxis.set_pane_color((0.0, 0.0,0.0, 0.0))            
  ax2.xaxis.set_pane_color((0.0, 0.0,0.0, 0.0))  
  ax2.yaxis.set_pane_color((0.0, 0.0,0.0, 0.0))  
  ax2.zaxis.set_pane_color((0.0, 0.0,0.0, 0.0))  
  #########################################
  return
############################################

#%%


def Plot_State(state,OBJ_File,gamma,Mesh_color,M_line=0.5,M_alpha=0.05):
    ##############################################################
    OBJ_Data = np.loadtxt(OBJ_File, delimiter=' ', dtype=str)
    
    # Extract vertex and face data
    vertices = np.array([line[1:].astype(float) for line in OBJ_Data if line[0] == 'v'])
    faces = np.array([line[1:].astype(int) for line in OBJ_Data if line[0] == 'f'])
    # Scale vertices
    vertices = vertices * gamma
    # Set faces to start at index 0 instead of 1
    faces = faces - 1
    # Create mesh
    mesh = Poly3DCollection([vertices[ii] for ii in faces], 
                            edgecolor=Mesh_color,
                            facecolors="white",
                            linewidth=M_line,
                            alpha=M_alpha)
    X = state[0,:]
    Y = state[1,:]  
    Z = state[2,:]
    #######################################################
    #######################################################
    figure3D = plt.figure()
    ax = figure3D.add_subplot(111, projection='3d')
    ax.add_collection3d(mesh)
    ax.plot(X, Y, Z, label='Trajectory',color='purple')
    ax.set_aspect('equal', 'box') 
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    return 

#%% Plot 4 by 4 


def plot_4by4(SimData):
    # Axis limit 
    # axes[1, 0].set_ylim([-0.05, 0.05])
    X = SimData['Position Vec. X']
    Y = SimData['Position Vec. Y']
    Z = SimData['Position Vec. Z']
    V_mag = np.sqrt(SimData['Velocity Vec. X']**2 + SimData['Velocity Vec. Y']**2 + SimData['Velocity Vec. Z']**2)
    # Corrected subplot creation
    fig, axes = plt.subplots(2, 2, figsize=(20, 20), subplot_kw={'projection': '3d'})
    axis0 = axes[0, 0]
    line_color ='#332288'
    ############################################################################
    # 3D Plot with V_mag as color
    scatter = axis0.scatter(X, Y, Z, c=V_mag, cmap='cool',s=0.1)
    cbar = fig.colorbar(scatter, ax=axis0, shrink=0.5, aspect=20)  
    cbar.set_label(r'$V (km/s)$', fontsize=25,labelpad=20) 
    axis0.view_init(elev=30, azim=45)   
    axis0.set_xlabel(r'$X (km)$', fontsize=25,labelpad=20)
    axis0.set_ylabel(r'$Y (km)$', fontsize=25,labelpad=20)
    axis0.set_zlabel(r'$Z (km)$', fontsize=25,labelpad=25)
    axis0.tick_params(axis='x', labelsize=20) 
    axis0.tick_params(axis='y', labelsize=20) 
    axis0.tick_params(axis='z', labelsize=20) 
    ############################################################################
    # Convert the rest of the axes back to 2D for the remaining plots
    for ax in axes.flat[1:]:
        ax.remove()
    axes[0, 1] = fig.add_subplot(2, 2, 2)
    axes[1, 0] = fig.add_subplot(2, 2, 3)
    axes[1, 1] = fig.add_subplot(2, 2, 4)
    # X-Y Plot
    axes[0, 1].plot(X, Y, color=line_color,linewidth=0.05)
    axes[0, 1].set_title("X-Y", fontsize=25)
    axes[0, 1].set_xlabel(r'$X (km)$', fontsize=25,labelpad=20)
    axes[0, 1].set_ylabel(r'$Y (km)$', fontsize=25,labelpad=20)
    axes[0, 1].tick_params(axis='x', labelsize=20) 
    axes[0, 1].tick_params(axis='y', labelsize=20) 
    ############################################################################
    # Y-Z Plot
    axes[1, 0].plot(Y, Z, color=line_color,linewidth=0.05)
    axes[1, 0].set_title("Y-Z", fontsize=25)
    axes[1, 0].set_xlabel(r'$Y (km)$', fontsize=25,labelpad=20)
    axes[1, 0].set_ylabel(r'$Z (km)$', fontsize=25,labelpad=20)
    axes[1, 0].tick_params(axis='x', labelsize=20) 
    axes[1, 0].tick_params(axis='y', labelsize=20) 
    ############################################################################
    # X-Z Plot
    axes[1, 1].plot(X, Z, color=line_color,linewidth=0.05)
    axes[1, 1].set_title("X-Z", fontsize=25)
    axes[1, 1].set_xlabel(r'$X (km)$', fontsize=25,labelpad=20)
    axes[1, 1].set_ylabel(r'$Z (km)$', fontsize=25,labelpad=20)
    axes[1, 1].tick_params(axis='x', labelsize=20) 
    axes[1, 1].tick_params(axis='y', labelsize=20) 
    ############################################################################
    plt.tight_layout()
    
    

#%% Triaxial Ellipsoid Gravity Potential Check
###########################################################################
#################################### Triaxial Ellipsoid Spherical Harmonics
def Triax_Ellip(Ax, x_in, Sphere_Override):
    """Triax_Ellip: Triaxial Ellipsoid Gravity Potential Check
        
        Calculates the gravitational potential of a triaxial ellipsoid
            - For numerically checking asteroid gravity calculations. 
            - Has options for sphere, where (alpha = beta = gamma).  
    
    Args:
        Ax (float): Body's largest semi-major axis (alpha_1) in kilometers (km).
        x_in (array): A linear space for input. Simulates a particle traveling away from the 
        Sphere_Override (string): S/E choice for sphere or triaxial ellipsoid calculations.

    Returns:
         U (list): The gravity potential of the ellipsoid/sphere. 
    """
    #############################################################
    ########################## Sphere Alpha, Beta, Gamma Override
    if Sphere_Override in ['S','s','sphere','Sphere','SPHERE']:
        alpha_Scale = 1 
        beta_Scale  = 1
        gamma_Scale = 1
    elif Sphere_Override in ['E','e','ellipse' ,'ellipsoid','ELLIPSE']:    
        alpha_Scale = 1 
        beta_Scale  = 0.5
        gamma_Scale = 0.3
    else:
        os.error()
    ####################################
    ########## Scale to semi-major axis
    alpha = Ax*alpha_Scale 
    beta  = Ax*beta_Scale
    gamma = Ax*gamma_Scale
    ################################
    ########## Define Potential list
    U = []
    #######################################################
    ##################### Begin Lööps ##################### 
    for i in x_in:
        x_0 = i #1.5 # 
        y_0 = 0
        z_0 = 0
        ### Set initial conditions array ###
        q = np.array([x_0 , y_0, z_0])
        r = np.linalg.norm(q)

        def elf(l):
            return (1 - (q[0]**2)/(alpha**2 + l) - (q[1]**2)/(beta**2 + l) - (q[2]**2)/(gamma**2 + l))
        

        lr = root(elf,0)
        li = lr.x[0]

        #rho1 = np.sqrt(q1[0]**2 + q1[1]**2 + q1[2]**2)


        RFe = elliprf(alpha**2+li,beta**2+li,gamma**2+li)
        RJalpha = elliprj(alpha**2+li,beta**2+li,gamma**2+li,alpha**2+li)
        RJbeta = elliprj(alpha**2+li,beta**2+li,gamma**2+li,beta**2+li)
        RJgamma = elliprj(alpha**2+li,beta**2+li,gamma**2+li,gamma**2+li)
        RJe = [RJalpha, RJbeta, RJgamma]


        # Bodies' potentials and total potential

        U_Calc = 1.5*RFe - 0.5*(q[0]**2*RJe[0] + q[1]**2*RJe[1] + q[2]**2*RJe[2])

        U.append(U_Calc)
    return np.array(U)
#####################################################
##################### End Lööps #####################


#%% Tetrahedron Volume Calculation
###############################################
def Tetra_Volume(Verts,Faces,MASCON_Div,total_vol=0):
    """Tetrahedron Volume Calculation 
            - for polyhedron shape models. 
    Args:
        Verts (array): Polyhedron Vertices
        Faces (array): Polyhedron Face Data
        MASCON_Div (fracton): Divides the tetrahedron, set = 1 for total
        total_vol (bool): 0/1 for total volume calculations, or  just tetra calculations
    """
    tetra_count=np.shape(Faces)[0]
    ###################    
    # Make list
    Volume_Tetra_Array  = np.zeros(tetra_count, dtype="float64")

    Tetra_U = np.zeros((tetra_count, 3), dtype="float64")
    Tetra_V = np.zeros((tetra_count, 3), dtype="float64")
    Tetra_W = np.zeros((tetra_count, 3), dtype="float64")
    
    ############################### Analytical Method:
    for it in range(0,tetra_count):
    ##### Center of mass to vertex vectors
        U_vec = np.array([Verts[Faces[it,0],0]*MASCON_Div,
                          Verts[Faces[it,0],1]*MASCON_Div,
                          Verts[Faces[it,0],2]*MASCON_Div
                        ]) 
        V_vec = np.array([Verts[Faces[it,1],0]*MASCON_Div,
                          Verts[Faces[it,1],1]*MASCON_Div,
                          Verts[Faces[it,1],2]*MASCON_Div
                        ]) 
        W_vec = np.array([Verts[Faces[it,2],0]*MASCON_Div,
                          Verts[Faces[it,2],1]*MASCON_Div,
                          Verts[Faces[it,2],2]*MASCON_Div
                        ])
        Tetra_U[it] = U_vec
        Tetra_V[it] = V_vec 
        Tetra_W[it] = W_vec 
        ######################################################
        ############### Triple Scalar Product ################
        Vol_Full_Sum =  U_vec[0]*(V_vec[1]*W_vec[2] - V_vec[2]*W_vec[1]) -\
                        U_vec[1]*(V_vec[0]*W_vec[2] - V_vec[2]*W_vec[0]) +\
                        U_vec[2]*(V_vec[0]*W_vec[1] - V_vec[1]*W_vec[0]) 
        #
        Vol_tetra_full = (1/6) * abs((Vol_Full_Sum))
        Volume_Tetra_Array[it] = Vol_tetra_full
    ###############################################
    ######### Sum Tetra Volumes for Check #########
    Total_Volume_Out = np.sum(Volume_Tetra_Array)
    #######################################################
    ######### Output ######################################
    if total_vol == 0:
        Volume_Tetra_Array_Message = f"""
{'-'*42}
|
|  Total Volume Calculated as:
|    V = {Total_Volume_Out}
|
{'-'*42}
"""
        # print(Volume_Tetra_Array_Message )
        return Volume_Tetra_Array,Total_Volume_Out
    elif total_vol == 1:
        return Volume_Tetra_Array











#%% condensed OBJ functions 

def OBJ_Read (OBJ_File):
    ##############################################################
    OBJ_Data = np.loadtxt(OBJ_File, delimiter=' ', dtype=str)
    
    # Extract vertex and face data
    vertices = np.array([line[1:].astype(float) for line in OBJ_Data if line[0] == 'v'])
    faces = np.array([line[1:].astype(int) for line in OBJ_Data if line[0] == 'f'])
    
    return vertices, faces


def OBJ_2_volInt(OBJ_File):
    """Reads an OBJ file and converts the vertex and face data into arrays.
    
    Args:
        OBJ_File (file): .obj file
    
    Returns:
        tuple: Two numpy arrays containing vertex and face data
    """
    # Read the OBJ file data
    OBJ_Data = np.loadtxt(OBJ_File, delimiter=' ', dtype=str)
    
    # Extract vertex data and prepend an empty row of zeros
    vertices = np.array([line[1:].astype(float) for line in OBJ_Data if line[0] == 'v'])
    vertices = np.vstack(([0, 0, 0], vertices))
    
    # Extract face data and prepend a 3 to each row
    faces = np.array([line[1:].astype(int) for line in OBJ_Data if line[0] == 'f'])
    faces = np.hstack((np.full((faces.shape[0], 1), 3), faces))
    
    return vertices, faces





#%%






