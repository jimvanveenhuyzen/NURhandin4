import numpy as np
import matplotlib.pyplot as plt
np.random.seed(121)

n_mesh = 16
n_part = 1024
positions = np.random.uniform(low=0, high=n_mesh, size=(3, n_part))

grid = np.arange(n_mesh) + 0.5
densities = np.zeros(shape=(n_mesh, n_mesh, n_mesh))
cellvol = 1.

for p in range(n_part):
    cellind = np.zeros(shape=(3, 2))
    dist = np.zeros(shape=(3, 2))

    for i in range(3):
        cellind[i] = np.where((abs(positions[i, p] - grid) < 1) |
                              (abs(positions[i, p] - grid - 16) < 1) | 
                              (abs(positions[i, p] - grid + 16) < 1))[0]
        dist[i] = abs(positions[i, p] - grid[cellind[i].astype(int)])

    cellind = cellind.astype(int)

    for (x, dx) in zip(cellind[0], dist[0]):    
        for (y, dy) in zip(cellind[1], dist[1]):
            for (z, dz) in zip(cellind[2], dist[2]):
                if dx > 15: dx = abs(dx - 16)
                if dy > 15: dy = abs(dy - 16)
                if dz > 15: dz = abs(dz - 16)

                densities[x, y, z] += (1 - dx)*(1 - dy)*(1 - dz) / cellvol
                
#PROBLEM 2A
                
#The following two lines are just delta = (rho - mean(rho))/mean(rho)
density_mean = 0.25 #1024 / 16**3#, more efficient to just write out 1/4th.
delta = (densities - density_mean) * 4 #dividing by 0.25 is multiplying by 4.

#The third coordinate of the grid is the z-coordinate, so convert 3D delta 
#into a 2d grid slice at z = 4.5,9.5,11.5,14.5
delta_z45 = delta[:,:,4]
delta_z95 = delta[:,:,9]
delta_z115 = delta[:,:,11]
delta_z145 = delta[:,:,14]

#Below are 4 colormap plots showing 2D grid slices of the density contrast at
#varying z-coordinates. Positive values indicate an overdensity, negative 
#values indicate underdensity. 
plt.imshow(delta_z45)
plt.colorbar()
plt.xlabel('x-coordinate')
plt.ylabel('y-coordinate')
plt.title(r'2D grid slice of density contrast $\delta$ at z=4.5')
plt.savefig('./2a_delta45.png')
plt.close()

plt.imshow(delta_z95)
plt.colorbar()
plt.xlabel('x-coordinate')
plt.ylabel('y-coordinate')
plt.title(r'2D grid slice of density contrast $\delta$ at z=9.5')
plt.savefig('./2a_delta95.png')
plt.close()

plt.imshow(delta_z115)
plt.colorbar()
plt.xlabel('x-coordinate')
plt.ylabel('y-coordinate')
plt.title(r'2D grid slice of density contrast $\delta$ at z=11.5')
plt.savefig('./2a_delta115.png')
plt.close()

plt.imshow(delta_z145)
plt.colorbar()
plt.xlabel('x-coordinate')
plt.ylabel('y-coordinate')
plt.title(r'2D grid slice of density contrast $\delta$ at z=14.5')
plt.savefig('./2a_delta145.png')
plt.close()

#PROBLEM 2B

from astropy import constants as const

#The only effect G has is multiplying the expression by a constant factor
#so the potential will look identical in a colormap plot regardless of our
#choice for the value of G. 

G = const.G.to_value() #gravitational constant in SI units

def FFT(x,N): #It is very important that N is a power of 2 and x a 1D array. 
    """
    An implementation of the recursive Cooley-Tukey algorithm. It only works if
    N is a power of 2 and if x is a 1 dimensional array. In the context of this
    problem N = 16 which a power of 2, so it works just fine. I found splitting
    the fft into an even and odd array and concatinating at the end the most
    intrusive. 
    """
    if N > 1:
        fft_even = FFT(x[::2],N/2) #split the data in two arrays
        fft_odd = FFT(x[1::2],N/2)
        for k in range(0,int(N/2)):
            t = fft_even[k]
            fft_even[k] = t + np.exp(2j*np.pi*k/N) * fft_odd[k] 
            fft_odd[k] = t - np.exp(2j*np.pi*k/N) * fft_odd[k]
        x = np.concatenate((fft_even,fft_odd)) #concatenate the arrays 
    return x #contains the final Fourier transformed array values

def invFFT(x,N):
    """
    An identical implementation of the recursive Cooley-Tukey algorithm, this
    time for an inverse Fourier transform. We take the complex conjugate of the
    exponent, which ends up being the only difference with the regular FFT.
    """
    if N > 1:
        ifft_even = invFFT(x[::2],N/2)
        ifft_odd = invFFT(x[1::2],N/2)
        for k in range(0,int(N/2)):
            t = ifft_even[k]
            ifft_even[k] = t + np.exp(-2j*np.pi*k/N) * ifft_odd[k]
            ifft_odd[k] = t - np.exp(-2j*np.pi*k/N) * ifft_odd[k]
        x = np.concatenate((ifft_even,ifft_odd))
    """
    The return value x contains the inverse Fourier transformed array values,
    multiplied by a factor N, so we must divide the outcome by a factor N to
    get the proper inversely transformed values. 
    """
    return x

def FFT_3D(x):
    """
    This is an implementation for a 3 dimensional grid. It Fourier transforms
    each possible '1D slice orientation' within the grid to obtain the result.
    3D FFT are simply 1D FFTs looped through various rows and columns. 
    """
    N = len(x)
    for i in range(N):
        for j in range(N):
            x[i,j,:] = FFT(x[i,j,:],N)
    for i in range(N):
        for j in range(N):
            x[:,i,j] = FFT(x[:,i,j],N)
    for i in range(N):
        for j in range(N):
            x[j,:,i] = FFT(x[j,:,i],N)
    return x 

def invFFT_3D(x):
    """
    Equivalent implementation of the 3D inverse FFT. Divides each iFFT by N as
    to properly normalise the result.
    """
    N = len(x)
    for i in range(N):
        for j in range(N):
            x[i,j,:] = invFFT(x[i,j,:],N)/N
    for i in range(N):
        for j in range(N):
            x[:,i,j] = invFFT(x[:,i,j],N)/N
    for i in range(N):
        for j in range(N):
            x[j,:,i] = invFFT(x[j,:,i],N)/N
    return x 

def k2_factor(x):
    """
    The wave-vector k is defined as (kx,ky,kz) = (nx,ny,nz)2pi/N. We use the 
    grid in the problem as the n vector (nx,ny,nz), as we evaluate the FFT
    at these points. A very important detail to remember is that we are
    operating on a grid with centres 0.5,1.5,etc., so we need to divide by
    these for kx,ky,kz rather than 0,1,2,etc. 
    """
    N = len(x) #amount of grid points per period/per wavelength 
    n_to_k = (N/(2*np.pi))**2 #converts from n to k space
    for i in range(N): #divide each delta point by correct kx,ky,kz squared
        for j in range(N):
            for k in range(N):
                x[i,j,k] = x[i,j,k]/((i+0.5)**2 + (j+0.5)**2 + (k+0.5)**2)
    return x*n_to_k #multiply by prefactor due to periodicity
    
def potential(delta):
    """
    I combine the whole conversion from Poisson to potential in this function
    """
    poisson = 4*np.pi*G*density_mean*(1+delta) #Poisson equation
    pot_fft = FFT_3D(poisson) #compute the FFT
    pot_fft = k2_factor(pot_fft) #convert to phi(k) \propto delta(k) / k^2
    return invFFT_3D(pot_fft) #perform final inverse FFT for potential phi(r)

grav_potential = potential(delta.astype(complex)) #need complex part for FFTs
grav_potential = grav_potential.astype(float)#removes (irrelevant) complex part

potential_z45 = grav_potential[:,:,4] #take the 4 2D slices as before
potential_z95 = grav_potential[:,:,9]
potential_z115 = grav_potential[:,:,11]
potential_z145 = grav_potential[:,:,14]

#potential plots for 2b
plt.imshow(potential_z45)
plt.colorbar()
plt.xlabel('x-coordinate')
plt.ylabel('y-coordinate')
plt.title(r'2D slice of potential $\Phi$ at z=4.5')
plt.savefig('./2b_pot45.png')
plt.close()

plt.imshow(potential_z95)
plt.colorbar()
plt.xlabel('x-coordinate')
plt.ylabel('y-coordinate')
plt.title(r'2D slice of potential $\Phi$ at z=9.5')
plt.savefig('./2b_pot95.png')
plt.close()

plt.imshow(potential_z115)
plt.colorbar()
plt.xlabel('x-coordinate')
plt.ylabel('y-coordinate')
plt.title(r'2D slice of potential $\Phi$ at z=11.5')
plt.savefig('./2b_pot115.png')
plt.close()

plt.imshow(potential_z145)
plt.colorbar()
plt.xlabel('x-coordinate')
plt.ylabel('y-coordinate')
plt.title(r'2D slice of potential $\Phi$ at z=14.5')
plt.savefig('./2b_pot145.png')
plt.close()

grav_potential = np.abs(grav_potential) #take absolute value of potential

potential_z45 = grav_potential[:,:,4] #take the 4 2D slices as before
potential_z95 = grav_potential[:,:,9]
potential_z115 = grav_potential[:,:,11]
potential_z145 = grav_potential[:,:,14]

#Plot colormaps of the log10 of the absolute values of the potential slices.
plt.imshow(np.log10(potential_z45))
plt.colorbar()
plt.xlabel('x-coordinate')
plt.ylabel('y-coordinate')
plt.title(r'2D slice of potential log$_{10}(|\Phi|)$ at z=4.5')
plt.savefig('./2b_log45.png')
plt.close()

plt.imshow(np.log10(potential_z95))
plt.colorbar()
plt.xlabel('x-coordinate')
plt.ylabel('y-coordinate')
plt.title(r'2D slice of potential log$_{10}(|\Phi|)$ at z=9.5')
plt.savefig('./2b_log95.png')
plt.close()

plt.imshow(np.log10(potential_z115))
plt.colorbar()
plt.xlabel('x-coordinate')
plt.ylabel('y-coordinate')
plt.title(r'2D slice of potential log$_{10}(|\Phi|)$ at z=11.5')
plt.savefig('./2b_log115.png')
plt.close()

plt.imshow(np.log10(potential_z145))
plt.colorbar()
plt.xlabel('x-coordinate')
plt.ylabel('y-coordinate')
plt.title(r'2D slice of potential log$_{10}(|\Phi|)$ at z=14.5')
plt.savefig('./2b_log145.png')
plt.close()



    


