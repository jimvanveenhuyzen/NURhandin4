import numpy as np
import matplotlib.pyplot as plt

features = np.genfromtxt("galaxy_data.txt",usecols=(0,1,2,3)) #m x n features
print(features)

kappa = features[:,0]
color = features[:,1]
extend = features[:,2]
flux = features[:,3]

def feature_scaling(x):
    return (x - np.mean(x))/np.std(x)

def sigmoid(z):
    return 1/(1+np.exp(-z))

kappa = feature_scaling(kappa)
color = feature_scaling(color)
extend = feature_scaling(extend)
flux = feature_scaling(flux)

features = np.array([kappa,color,extend,flux])
features = np.transpose(features)
print(features)

plt.hist(kappa,edgecolor='black',bins=np.linspace(-2,2,10))
plt.xlabel('Rescaled value of Kappa_co')
plt.ylabel('Counts inside respective bin')
plt.show()

plt.hist(color,edgecolor='black')
plt.xlabel('Rescaled value of the color')
plt.ylabel('Counts inside respective bin')
plt.show()

plt.hist(extend,edgecolor='black',bins=np.linspace(-0.3,0.25,10))
plt.xlabel('Rescaled value of the galaxy extension')
plt.ylabel('Counts inside respective bin')
plt.show()

plt.hist(flux,edgecolor='black',bins=np.linspace(-0.02,0.045,10))
plt.xlabel('Rescaled value of the emission line flux')
plt.ylabel('Counts inside respective bin')
plt.show()

