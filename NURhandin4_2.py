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
                
density_mean = 0.25 #1024 / 16**3, more efficient to just write out 1/4th.

delta = (densities - density_mean) * 4 #dividing by 0.25 is multiplying by 4.

delta_z45 = delta[:][:][4]
delta_z95 = delta[:][:][9]
delta_z115 = delta[:][:][11]
delta_z145 = delta[:][:][14]

plt.imshow(delta_z45)
plt.colorbar()
plt.xlabel('x-coordinate')
plt.ylabel('y-coordinate')
plt.title('2D grid slice of the density contrast at z=4.5')
plt.show()

plt.imshow(delta_z95)
plt.colorbar()
plt.xlabel('x-coordinate')
plt.ylabel('y-coordinate')
plt.title('2D grid slice of the density contrast at z=9.5')
plt.show()

plt.imshow(delta_z115)
plt.colorbar()
plt.xlabel('x-coordinate')
plt.ylabel('y-coordinate')
plt.title('2D grid slice of the density contrast at z=11.5')
plt.show()

plt.imshow(delta_z145)
plt.colorbar()
plt.xlabel('x-coordinate')
plt.ylabel('y-coordinate')
plt.title('2D grid slice of the density contrast at z=14.5')
plt.show()

#PROBLEM 2B

from astropy import constants as const

G = const.G.to_value()

def FFT(x,N): #N needs to be a power of 2. 
    if N > 1:
        fft_even = FFT(x[::2],N/2)
        fft_odd = FFT(x[1::2],N/2)
        for k in range(0,int(N/2)):
            t = fft_even[k]
            fft_even[k] = t + np.exp(2j*np.pi*k/N) * fft_odd[k] 
            fft_odd[k] = t - np.exp(2j*np.pi*k/N) * fft_odd[k]
        x = np.concatenate((fft_even,fft_odd))
    return x

def invFFT(x,N):
    if N > 1:
        ifft_even = invFFT(x[::2],N/2)
        ifft_odd = invFFT(x[1::2],N/2)
        for k in range(0,int(N/2)):
            t = ifft_even[k]
            ifft_even[k] = t + np.exp(-2j*np.pi*k/N) * ifft_odd[k]
            ifft_odd[k] = t - np.exp(-2j*np.pi*k/N) * ifft_odd[k]
        x = np.concatenate((ifft_even,ifft_odd))
    return x

def FFT_3D(x):
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

print(delta.shape)

def potential(delta):
    poisson = 4*np.pi*G*density_mean*(1+delta)
    pot_fft = FFT_3D(poisson)
    k = len(delta)
    pot_fft = pot_fft * k**(-2)
    return invFFT_3D(pot_fft)

def potentialnp(delta): #is the same as my method! 
    poisson = 4*np.pi*G*density_mean*(1+delta)
    pot_fft = np.fft.fftn(poisson)
    k = len(delta)
    pot_fft = pot_fft * k**(-2)
    return np.fft.ifftn(pot_fft)

#print('own method\n',potential(delta.astype(complex)))
#print('numpy method\n',potentialnp(delta.astype(complex)))

grav_potential = potential(delta.astype(complex))

#We apply 2 filters, first we take the absolute value of the potential, as 
#negative potential does not exist.
#Next, we set values that are very small (smaller than 10^-18) to some lower
#limit on the potential. This is to 1) avoid zeros, which are singularities 
#because the potential cannot be 0, and 2) to better display the contrast 
#between relevant density values in the range 10^-12 to 10^-16. 

grav_potential = grav_potential.astype(float)

potential_z45 = grav_potential[:][:][4]
potential_z95 = grav_potential[:][:][9]
potential_z115 = grav_potential[:][:][11]
potential_z145 = grav_potential[:][:][14]

#potential plots

plt.imshow(potential_z45)
plt.colorbar()
plt.xlabel('x-coordinate')
plt.ylabel('y-coordinate')
plt.title('2D grid slice of the potential at z=4.5')
plt.show()

plt.imshow(potential_z95)
plt.colorbar()
plt.xlabel('x-coordinate')
plt.ylabel('y-coordinate')
plt.title('2D grid slice of the potential at z=9.5')
plt.show()

plt.imshow(potential_z115)
plt.colorbar()
plt.xlabel('x-coordinate')
plt.ylabel('y-coordinate')
plt.title('2D grid slice of the potential at z=11.5')
plt.show()

plt.imshow(potential_z145)
plt.colorbar()
plt.xlabel('x-coordinate')
plt.ylabel('y-coordinate')
plt.title('2D grid slice of the potential at z=14.5')
plt.show()

grav_potential = np.where(grav_potential > 1.0e-18, grav_potential, 1.0e-18)

potential_z45 = grav_potential[:][:][4]
potential_z95 = grav_potential[:][:][9]
potential_z115 = grav_potential[:][:][11]
potential_z145 = grav_potential[:][:][14]

#log10 of the potential

plt.imshow(np.log10(potential_z45))
plt.colorbar()
plt.xlabel('x-coordinate')
plt.ylabel('y-coordinate')
plt.title('2D grid slice of the potential at z=4.5')
plt.show()

plt.imshow(np.log10(potential_z95))
plt.colorbar()
plt.xlabel('x-coordinate')
plt.ylabel('y-coordinate')
plt.title('2D grid slice of the potential at z=9.5')
plt.show()

plt.imshow(np.log10(potential_z115))
plt.colorbar()
plt.xlabel('x-coordinate')
plt.ylabel('y-coordinate')
plt.title('2D grid slice of the potential at z=11.5')
plt.show()

plt.imshow(np.log10(potential_z145))
plt.colorbar()
plt.xlabel('x-coordinate')
plt.ylabel('y-coordinate')
plt.title('2D grid slice of the potential at z=14.5')
plt.show()



    


