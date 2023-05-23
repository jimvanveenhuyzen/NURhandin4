import numpy as np
import matplotlib.pyplot as plt

#PROBLEM 3A

data = np.genfromtxt("galaxy_data.txt") #m x n features
print(data)

#extract the four features from the data to re-scale and use in minimization
#I will refer to the features in order of 0,1,2,3 in variable names! 
kappa = data[:,0]
color = data[:,1]
extend = data[:,2]
flux = data[:,3]

ylabel = data[:,4]

def feature_scaling(x):
    """
    Simple function to scale some feature of m objects such that it has mean 0
    and std 1. 
    """
    return (x - np.mean(x))/np.std(x)

def sigmoid(z):
    """
    Implementation of the sigmoid function
    """
    return 1/(1+np.exp(-z))

kappa_scaled = feature_scaling(kappa)
color_scaled = feature_scaling(color)
extend_scaled = feature_scaling(extend)
flux_scaled = feature_scaling(flux)

plt.hist(kappa_scaled,edgecolor='black',bins=np.linspace(-2,2,10))
plt.xlabel(r'Rescaled value of $\kappa_{co}$')
plt.ylabel('Counts inside respective bin')
plt.show()

plt.hist(color_scaled,edgecolor='black')
plt.xlabel('Rescaled value of the color')
plt.ylabel('Counts inside respective bin')
plt.show()

plt.hist(extend_scaled,edgecolor='black',bins=np.linspace(-0.3,0.25,10))
plt.xlabel('Rescaled value of the galaxy extension')
plt.ylabel('Counts inside respective bin')
plt.show()

plt.hist(flux_scaled,edgecolor='black',bins=np.linspace(-0.02,0.045,10))
plt.xlabel('Rescaled value of the emission line flux')
plt.ylabel('Counts inside respective bin')
plt.show()

#PROBLEM 3B

def cost_function(xdata,ydata,weights,epsilon):
    m = len(xdata) #m objects per feature
    z = np.matmul(weights,np.transpose(xdata))
    h = sigmoid(z)
    J = ydata * np.log(h+epsilon) + (1-ydata) * np.log(1-h+epsilon)
    return -np.sum(J)/m

def gradient_descent(xdata,ydata,weights,alpha,epsilon,iterations):
    m = len(xdata)
    J_values = np.zeros(iterations)
    for j in range(iterations):
        z = np.matmul(weights,np.transpose(xdata)) #theta (weights) times x^T 
        h = sigmoid(z) #hypothesis
        weights = weights - alpha * np.matmul(np.transpose(xdata), (h-ydata))/m
        J = cost_function(xdata,ydata,weights,epsilon)
        J_values[j] = J
    return J_values, weights

#Try minimization for all the features at once to see which features have 
#the most weight as a reference for pairing features together and interpreting
#the results

features = np.array([kappa_scaled,color_scaled,extend_scaled,flux_scaled])
features = np.transpose(features)
print(features)

weights_init = np.ones(len(features[1]))

minimized_J, minimized_weights =\
    gradient_descent(features,ylabel,weights_init,0.05,1e-6,20000)
print('The minimized weight values are',minimized_weights)

plt.plot(np.arange(0,20000,1),minimized_J)
plt.xlabel('Number of iterations N')
plt.ylabel(r'Cost function J($\theta$)')
plt.show()

#Try minimization for combination color (feature 1) and extend (feature 2)

features_12 = np.array([color_scaled,extend_scaled])
features_12 = np.transpose(features_12)
weights_12 = np.ones(len(features_12[1]))

min_J_12, min_weights_12 =\
    gradient_descent(features_12,ylabel,weights_12,0.05,1e-6,5000)
print('The minimized weight values for color and extend are',min_weights_12)
                     
plt.plot(np.arange(0,5000,1),min_J_12)
plt.xlabel('Number of iterations N')
plt.ylabel(r'Cost function J($\theta$)')
plt.show()

#Try minimization for combination kappa_co (feature 0) and extend (feature 1)

features_01 = np.array([kappa_scaled,color_scaled])
features_01 = np.transpose(features_01)
weights_01 = np.ones(len(features_01[1]))

min_J_01, min_weights_01 =\
    gradient_descent(features_01,ylabel,weights_01,0.05,1e-6,5000)
print('The minimized weight values for kappa_co and extend are',min_weights_01)
                     
plt.plot(np.arange(0,5000,1),min_J_01)
plt.xlabel('Number of iterations N')
plt.ylabel(r'Cost function J($\theta$)')
plt.show()

#Try minimization for combination extend (feature 2) and line-flux (feature 3)

features_23 = np.array([extend_scaled,flux_scaled])
features_23 = np.transpose(features_23)
weights_23 = np.ones(len(features_23[1]))

min_J_23, min_weights_23 =\
    gradient_descent(features_23,ylabel,weights_23,0.01,1e-6,5000)
print('The minimized weight values for extend and lineflux are',min_weights_23)
                     
plt.plot(np.arange(0,5000,1),min_J_23)
plt.xlabel('Number of iterations N')
plt.ylabel(r'Cost function J($\theta$)')
plt.show()

#PROBLEM 3C

def y_model(xdata,model_weights):
    m = len(xdata) #m objects per feature 
    y_model = np.zeros(m)
    for i in range(m):
        z = np.matmul(model_weights,np.transpose(xdata[i]))
        y_model[i] = sigmoid(z)
    y_model = np.where(y_model < 0.5, 0, 1) 
    return y_model

ymodel = y_model(features,minimized_weights)
#print('model rescaled',ymodel_y0)
#print('data',ylabel)

def confusion_matrix(ydata,y_model):
    m = len(ydata) #number of objects
    CM_11, CM_12, CM_21, CM_22 = 0,0,0,0
    for i in range(m):
        if y_model[i] == 1 and ydata[i] == 1: #true positive
            CM_11 += 1
        if y_model[i] - ydata[i] == 1: #false positive
            CM_12 += 1
        elif y_model[i] - ydata[i] == -1: #false negative
            CM_21 += 1 
        else: #true negative: ymodel_i and ydata_i are both 0
            CM_22 += 1
    return np.array([[CM_11,CM_12],[CM_21,CM_22]])

CM_all = confusion_matrix(ylabel,ymodel)

print('The confusion matrix of all features is given by\n',CM_all)

def F_score(beta,CM): #input a 2x2 confusion matrix and a beta
    TP, FP, FN = CM[0,0],CM[0,1],CM[1,0] #get true pos, false pos and false neg
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    return (1 + beta**2) * (precision * recall)/((beta**2)*precision + recall)

F1_score = F_score(1,CM_all)
print('The F1 score of the training set using all features is',F1_score)

#try features color (1) and extend (2) as a combination
ymodel_12 = y_model(features_12,min_weights_12)
CM_features12 = confusion_matrix(ylabel,ymodel_12)
print('The confusion matrix for color and extend is given by\n',CM_features12)
F1_score_12 = F_score(1,CM_features12)
print('The F1 score of the training set using color and extend is',F1_score_12)

#try features kappa_co (0) and color (1) as a combination
ymodel_01 = y_model(features_01,min_weights_01)
CM_features01 = confusion_matrix(ylabel,ymodel_01)
print('The confusion matrix for kappa and color is given by\n',CM_features01)
F1_score_01 = F_score(1,CM_features01)
print('The F1 score of the training set using kappa and color is',F1_score_01)

#try features extend (2) and line flux (3) as a combination
ymodel_23 = y_model(features_23,min_weights_23)
CM_features23 = confusion_matrix(ylabel,ymodel_23)
print('The confusion matrix for extend and flux is given by\n',CM_features23)
F1_score_23 = F_score(1,CM_features23)
print('The F1 score of the training set using extend and flux is',F1_score_23)
    





