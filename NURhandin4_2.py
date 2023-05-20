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
plt.title('2D grid slice at z=4.5')
plt.show()

plt.imshow(delta_z95)
plt.colorbar()
plt.xlabel('x-coordinate')
plt.ylabel('y-coordinate')
plt.title('2D grid slice at z=9.5')
plt.show()

plt.imshow(delta_z115)
plt.colorbar()
plt.xlabel('x-coordinate')
plt.ylabel('y-coordinate')
plt.title('2D grid slice at z=11.5')
plt.show()

plt.imshow(delta_z145)
plt.colorbar()
plt.xlabel('x-coordinate')
plt.ylabel('y-coordinate')
plt.title('2D grid slice at z=14.5')
plt.show()

#PROBLEM 2B

def FFT(x,N):
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
        fft_even = FFT(x[::2],N/2)
        fft_odd = FFT(x[1::2],N/2)
        for k in range(0,int(N/2)):
            t = fft_even[k]
            fft_even[k] = t + np.exp(-2j*np.pi*k/N) * fft_odd[k] 
            fft_odd[k] = t - np.exp(-2j*np.pi*k/N) * fft_odd[k]
        x = np.concatenate((fft_even,fft_odd))
    return x

def FFT_2D(x): #input a 2 dimensional array
    N = len(x)
    for i in range(N):
        x[i] = FFT(x[i],N)
    x = np.transpose(x)
    for i in range(N):
        x[i] = FFT(x[i],N)
    return x

delta_z45_FFT = FFT_2D(delta_z45.astype(complex))
print(delta_z45_FFT)


