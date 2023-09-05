import numpy as np
import matplotlib.pyplot as plt
import itertools 

EPS = 1e-9 #precision parameter

def matrix_equals(U,V):
    '''Returns true if U and V are (approximately) equal    
    '''
    diff = np.linalg.norm(U - V, 'fro')
    if diff < EPS:
        return True
    else:
        return False

def is_orthonormal(U):
    '''Returns True if U is orthogonal else returns False'''
    if matrix_equals(np.matmul(U,U.T), np.identity(U.shape[0])) and \
        matrix_equals(np.matmul(U.T,U), np.identity(U.shape[0])): 
        return True
    else:
        return False 


    
def visualize_function (U = np.identity(2), D = np.ones(2), VT = np.identity(2), num_grid_points_per_dim = 200,\
               disable_checks = False, show_original = True, show_VT = True, show_DVT = True, show_UDVT = True):
   
    if disable_checks is False:
        #Type checks
        if not isinstance(U, np.ndarray):
            raise ValueError('U must be a np.ndarray')
        if not isinstance(D, np.ndarray):
            raise ValueError('D must be a np.ndarray')
        if not isinstance(VT, np.ndarray):
            raise ValueError('VT must be a np.ndarray')
        if not isinstance(num_grid_points_per_dim, int):
            raise ValueError('num_grid_points_per_dim must be an integer')

        #Dimension checks

        if len(U.shape) != 2 or (U.shape != np.array([2,2])).any():
            raise ValueError('U must have shape [2,2]')   
        if len(D.shape) != 1 or (D.shape != np.array([2])).any():
            raise ValueError('D must have shape [2,]')    
        if len(VT.shape) != 2 or (VT.shape != np.array([2,2])).any():
            raise ValueError('VT must have shape [2,2]')

        #Sanity checks
        if not is_orthonormal(U):
            raise ValueError('U must be an orthogonal matrix')
        if not is_orthonormal(VT):
            raise ValueError('VT must be an orthogonal matrix')

        #Compute check
        if num_grid_points_per_dim >= 300:
            raise ValueError('num_grid_points_per_dim must be less than or equal to 300 to not overload memory')
    
    
    #Define unit circle here
    x = np.linspace(-1, 1, num_grid_points_per_dim)
    y = np.linspace(-1, 1, num_grid_points_per_dim)
    xy_grid = np.array(list(itertools.product(x,y)))
    xy_unit_circle = xy_grid[xy_grid[:,0]**2 + xy_grid[:,1]**2 < 1, :]
       
    
    #Define basis vectors for x and y axis    
    e_x = np.vstack([np.linspace(0,1,num_grid_points_per_dim), np.zeros(num_grid_points_per_dim)]).T
    e_y = np.vstack([np.zeros(num_grid_points_per_dim), np.linspace(0,1,num_grid_points_per_dim)]).T
       
       
    
    #Show the original image here 
    if show_original is True:
        plt.scatter(xy_unit_circle[:,0], xy_unit_circle[:,1], c = 'lightblue', s = 5)
        plt.scatter(e_x[:,0], e_x[:,1], c = 'red', s = 5)
        plt.scatter(e_y[:,0], e_y[:,1], c = 'green', s = 5)
        plt.title('Unit circle')
        plt.axis('equal')
        plt.show()
    
    
    #Apply V_T transformation here
    VT_unit_circle = np.matmul(VT,xy_unit_circle.T).T
    VT_e_x = np.matmul(VT,e_x.T).T
    VT_e_y = np.matmul(VT, e_y.T).T
    
    if show_VT is True:
        plt.scatter(VT_unit_circle[:,0], VT_unit_circle[:,1], c = 'lightblue', s = 5)
        plt.scatter(VT_e_x[:,0], VT_e_x[:,1], c = 'red', s = 5)
        plt.scatter(VT_e_y[:,0], VT_e_y[:,1], c = 'green', s = 5)
        plt.title('Unit circle transformed by V_T')
        plt.axis('equal')
        plt.xlim([-1.5, 1.5])
        plt.show()

    
    #Apply D transformation here
    Dm = np.diag(D)
    DVT_unit_circle = np.matmul(Dm,VT_unit_circle.T).T
    DVT_e_x = np.matmul(Dm,VT_e_x.T).T
    DVT_e_y = np.matmul(Dm, VT_e_y.T).T
    
        
    if show_DVT is True:
        plt.scatter(DVT_unit_circle[:,0], DVT_unit_circle[:,1], c = 'lightblue', s = 5)
        plt.scatter(DVT_e_x[:,0], DVT_e_x[:,1], c = 'red', s = 5)
        plt.scatter(DVT_e_y[:,0], DVT_e_y[:,1], c = 'green', s = 5)
        plt.title('Unit circle transformed by DV_T')
        plt.axis('equal')
        plt.show()

    
    #Apply U transformation here
    UDVT_unit_circle = np.matmul(U,DVT_unit_circle.T).T
    UDVT_e_x = np.matmul(U,DVT_e_x.T).T
    UDVT_e_y = np.matmul(U, DVT_e_y.T).T
    
    if show_UDVT is True:
        plt.scatter(UDVT_unit_circle[:,0], UDVT_unit_circle[:,1], c = 'lightblue', s = 5)
        plt.scatter(UDVT_e_x[:,0], UDVT_e_x[:,1], c = 'red', s = 5)
        plt.scatter(UDVT_e_y[:,0], UDVT_e_y[:,1], c = 'green', s = 5)
        plt.title('Unit circle transformed by UDV_T')
        plt.axis('equal')
        plt.show()
