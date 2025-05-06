# Based on InputGen.f90 from poro_nondimensional.f90
# Translated to python by ejf61
# Can be run manually or imported into other python scripts
# Running manually is interactive & the default values have to be changed in the main() function. Saves the output
# If calling from another script, perm isn't saved. Example for running:
# import generate_permeability as gp
# perm = gp.uniform(nx,ny)

# Packages 
import numpy as np

output_dir = '/mnt/c/Users/Emily/Documents/PhD/Scripts/Python/poro_current/Input/'

def main():
    # The main() function is v. out of date. Not using atm, but if need to in future, it will need some work.
    nx = 120
    ny = 120
    dx = 0.5 # Used for sin_cheq and chequerboard. Set to 1 to return to grid-based generation
    dy = 0.05
    k_a = 1
    k_b = 0.01
    
    print('Choose a permeability distribution:')
    choice = int(input('1)Uniform 2)Low-perm rectangle 3)High-perm channel 4)Chequerboard 5) Sinusoidal chequerboard 6) Sinusoidal lines \n'))
    
    if choice == 1:
        perm = uniform(nx,ny)
    
    elif choice == 2:
        perm = rectangle(nx,ny,k_a,k_b)
        
    elif choice == 3:
        perm = channel(nx,ny,k_a,k_b)
    
    elif choice == 4:
        print('Choose the horizontal and vertical dimensions of the chequerboard')
        perm_dx = int(input('Horizontal:\n'))
        perm_dy = int(input('Vertical:\n'))
        print(f'Chequerboard values are: {perm_dx}, {perm_dy}')
        perm = chequerboard(nx,ny,dx,dy,k_a,k_b,perm_dx,perm_dy)
    
    elif choice == 5:
        print('Choose the horizontal and vertical dimensions of the chequerboard')
        # Should be divisible by 4 to get sin = +/- 1
        perm_dx = int(input('Horizontal:\n'))
        perm_dy = int(input('Vertical:\n'))
        print(f'Chequerboard values are: {perm_dx}, {perm_dy}')
        perm_min = min(k_a,k_b)
        perm_mean = 1
        print('Do you want to inject at a maximum or minimum?')
        choice2 = int(input('1)Maximum 2)Minimum\n'))
        if choice2 == 1:
            inj_max = True
        else:
            inj_max = False
        
        perm = sin_cheq(nx,ny,dx,dy,perm_mean,perm_min,perm_dx,perm_dy,inj_max)
        
    elif choice==6:
        print('Choose the wavelength of the y perm variations:\n')
        perm_dy = int(input('perm_dy:\n'))
        perm_min = min(k_a,k_b)
        perm_mean = 1
        perm = sin_lines(nx,ny,dx,dy,perm_mean,perm_min,perm_dy)
    else:
        print('Incorrect choice, please try again')
    
    np.savetxt(output_dir + './Permeability.txt', perm)

def gen_perm(nx,ny,dx,dy,perm_type,perm_mean,perm_ampl,perm_wly):
    perm_type = perm_type.lower()
    
    if perm_type == 'uniform':
        perm = uniform(nx,ny)
    
    elif perm_type == 'cos':
        perm = sin_lines(nx,ny,dx,dy,perm_mean,perm_ampl,perm_wly)
        
    elif perm_type == 'gauss':
        perm = gauss(nx,ny,dy,perm_mean[0],perm_ampl[0],perm_wly[0])
    
    elif perm_type == 'channel':
        perm = channel(nx,ny,dy,perm_mean[0],perm_ampl[0],perm_wly[0])
    
    else:
        perm = np.zeros((nx,ny)) # This will force an error and poro_current will quit.
    
    return perm

def uniform(nx,ny):
    """
    (nx,ny)
    Creates uniform permeability field size (nx,ny)
    """
    perm = np.ones((nx,ny))
    return perm
    
def rectangle(nx,ny,k_a,k_b):
    """
    (nx,ny,k_a,k_b)
    Creates perm field of value k_a with rectangle of k_b
    """
    # Should add size of rectangle as an input
    perm = np.zeros((nx,ny))
    perm[:,:] = k_a
    x_start = int(nx/2)-20
    x_end = int(nx/2)+20
    y_start = int(ny/2)+10
    y_end = int(ny/2)+30
    # Not sure this indexing will work - might need to meshgrid
    # Revisit if ever want a rectangle
    perm[x_start:x_end,y_start:y_end] = k_b
    return perm

def channel(nx,ny,dy,perm_mean,perm_ampl,perm_wly):
    """
    (nx,ny,dy,perm_mean,perm_ampl,perm_wly)
    Creates perm field of perm_mean with channel of perm_mean+perm_ampl,
    centred on middle of y range with width perm_wly
    """
    y_tmp = np.arange(-ny//2,ny//2) * dy # This is simple way to centre channel
    ind_channel = np.argwhere(np.abs(y_tmp)<=perm_wly/2)
    
    perm = np.ones((nx,ny)) * perm_mean
    perm[:,ind_channel] = perm_mean + perm_ampl
    return perm

def gauss(nx,ny,dy,perm_mean,perm_ampl,perm_wly):
    """
    (nx,ny,dy,perm_mean,perm_ampl,perm_wly)
    Creates perm field of perm_mean with gaussian of amplitdue perm_ampl,
    centred on middle of y range with width perm_wly
    """
    y_tmp = np.arange(-ny//2,ny//2) * dy # This is simple way to centre gaussian
    perm = np.zeros((nx,ny))
    perm[:] = perm_mean + perm_ampl * np.exp(-(2*y_tmp/perm_wly)**2)
    return perm
    

def chequerboard(nx,ny,dx,dy,k_a,k_b,perm_dx,perm_dy):
    """
    (nx,ny,k_a,k_b,perm_dx,perm_dy)
    Creates a chequerboard permeability field
    perm_dx, perm_dy are size of chequerboard squares (in non-diml length)
    k_a is top left value
    """
    perm = np.zeros((nx,ny))
    
    CBdx = int(perm_dx/dx)
    CBdy = int(perm_dy/dy)
    
    # Injecting at 1,1
    # Want injection pt at centre of a chequerboard block
    # Deal with first row/column differently
    
    CBdx_init = int(np.ceil(CBdx/2)) + 1 # The +1 accounts for injecting at 1,1
    CBdy_init = int(np.ceil(CBdy/2)) + 1
    
    # Number of 'normal' chequerboard blocks
    rx = int(np.ceil( ((nx+1) - CBdx_init - 1) / CBdx))
    ry = int(np.ceil( ((ny+1) - CBdy_init - 1) / CBdy))
    
    # Top left block
    # always k_a
    perm[0:CBdx_init, 0:CBdy_init] = k_a
    
    # Bulk of top row
    for i in range(0, rx-1):
        #  k_a if i+1 even, k_b if odd
        perm[CBdx_init+i*CBdx:CBdx_init+(i+1)*CBdx, 0:CBdy_init] = 0.5*(k_a+k_b) + ((-1)**(i+1)) * 0.5*(k_a-k_b)
    
    # Top right block
    # k_a if rx even, k_b if odd 
    perm[CBdx_init+(rx-1)*CBdx:nx,0:CBdy_init] = 0.5*(k_a+k_b) + ((-1)**(rx)) * 0.5*(k_a-k_b)
    
    # Bulk of left column
    for j in range(0, ry-1):
        # k_a if j+1 even, k_b if odd
        perm[0:CBdx_init, CBdy_init+j*CBdy:CBdy_init+(j+1)*CBdy] = 0.5*(k_a+k_b) + ((-1)**(j+1)) * 0.5*(k_a-k_b)
    
    # Bottom left block
    # k_a if ry even, k_b if odd
    perm[0:CBdx_init, CBdy_init+(ry-1)*CBdy:ny] = 0.5*(k_a+k_b) + ((-1)**(ry)) * 0.5*(k_a-k_b)
    
    # 'Normal' chequerboard blocks
    # Start from CBdx_init, CBdy_init
    for j in range(0, ry-1):
        for i in range(0, rx-1):
            # k_a if i+j even, k_b if odd
            perm[CBdx_init+i*CBdx:CBdx_init+(i+1)*CBdx, CBdy_init+j*CBdy:CBdy_init+(j+1)*CBdy] = 0.5*(k_a+k_b) + ((-1)**((i + j))) * 0.5*(k_a-k_b)
            
        # Likely to be some left over space at right/bottom
        # Bulk of right column
        # k_a if j+rx-1 is even, k_b if odd
        perm[CBdx_init+(rx-1)*CBdx:nx, CBdy_init+j*CBdy:CBdy_init+(j+1)*CBdy] = 0.5*(k_a+k_b) + ((-1)**(( j + rx -1 ))) * 0.5*(k_a-k_b)
    
    # Bulk of bottom row
    for i in range(0, rx-1):
        # k_a if i+ry-1 is even, k_b if odd
        perm[CBdx_init+i*CBdx:CBdx_init+(i+1)*CBdx, CBdy_init+(ry-1)*CBdy:ny] = 0.5*(k_a+k_b) + ((-1)**((i + ry - 1))) * 0.5*(k_a-k_b)
    
    # Bottom right block
    # k_a if rx+ry is even, k_b if odd
    perm[CBdx_init+(rx-1)*CBdx:nx, CBdy_init+(ry-1)*CBdy:ny] = 0.5*(k_a+k_b) + ((-1)**(rx + ry)) * 0.5*(k_a-k_b)
    return perm

def sin_cheq(nx,ny,dx,dy,perm_mean,perm_ampl,perm_wlx,perm_wly,inj_max):
    """
    (nx,ny,dx,dy,perm_mean,perm_ampl,CBdx,CBdy)
    Creates a sinusoidal chequerboard
    perm_wlx,perm_wly are *distance* between peaks
    """    
    # Injection pt. Could/should be an input?
    Q_x = 1
    Q_y = 1
    
    # Set up grid
    x = (np.arange(nx) - Q_x)*dx
    y = (np.arange(ny) - Q_y)*dy
    X,Y = np.meshgrid(x,y)
    
    if inj_max:
        perm = (np.cos(2*np.pi/perm_wlx * X)+np.cos(2*np.pi/perm_wly * Y)) * perm_ampl + perm_mean
    else:
        perm = (np.cos(2*np.pi/perm_wlx * X+np.pi)+np.cos(2*np.pi/perm_wly * Y+np.pi)) * perm_ampl + perm_mean
    return perm

def sin_lines(nx,ny,dx,dy,perm_mean,perm_ampl,perm_wly):
    """
    (nx,ny,dx,dy,perm_mean,perm_ampl,perm_wly)
    Creates a multi-modal sinusoidal permeability field in y - constant in x
    Assumes perm_mean,perm_ampl,perm_wly are all the same length & each triplet defines a mode
    Works for N_modes=1 too
    """
    N_modes = len(perm_ampl)
    
    # Injection pt. Could/should be an input?
    Q_x = 1
    Q_y = 1
    
    # Set up grid
    #x = (np.arange(nx) - Q_x)*dx
    y = (np.arange(ny) - Q_y)*dy
    
    perm = np.zeros((nx,ny))
    if N_modes==1:
        perm[:] += perm_ampl[0] * np.cos(2*np.pi/perm_wly[0] * y - np.pi) + perm_mean[0]
    else:
        for i in range(N_modes):
            perm[:] += perm_ampl[i] * np.cos(2*np.pi/perm_wly[i] * y - np.pi) + perm_mean[i]/N_modes
    return perm

if __name__ == '__main__':
    main()
