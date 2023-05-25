import numpy as np
import matplotlib.pyplot as plt

#PROBLEM 3A

data = np.genfromtxt("galaxy_data.txt") #m x (n-1) features

#extract the four features from the data to re-scale and use in minimization
#I will refer to the features in order of 0,1,2,3 in variable names! 
kappa = data[:,0]
color = data[:,1]
extend = data[:,2]
flux = data[:,3]

ylabel = data[:,4] #the column representing the ydata we use to train the model

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

#Apply feature scaling on all n=4 features of the data
kappa_scaled = feature_scaling(kappa)
color_scaled = feature_scaling(color)
extend_scaled = feature_scaling(extend)
flux_scaled = feature_scaling(flux)

#plot the distributions of the rescaled features in histograms with custom bins
plt.hist(kappa_scaled,edgecolor='black',bins=np.linspace(-2,2,10))
plt.xlabel(r'Rescaled value of $\kappa_{co}$')
plt.ylabel('Counts inside respective bin')
plt.title(r'Distribution of the rescaled $\kappa_{co}$ feature')
plt.savefig('./3a_kappa.png')
plt.close()

plt.hist(color_scaled,edgecolor='black')
plt.xlabel('Rescaled value of the color')
plt.ylabel('Counts inside respective bin')
plt.title('Distribution of the rescaled color feature')
plt.savefig('./3a_color.png')
plt.close()

plt.hist(extend_scaled,edgecolor='black',bins=np.linspace(-0.3,0.25,10))
plt.xlabel('Rescaled value of the galaxy extension')
plt.ylabel('Counts inside respective bin')
plt.title('Distribution of the rescaled galaxy extension feature')
plt.savefig('./3a_extend.png')
plt.close()

plt.hist(flux_scaled,edgecolor='black',bins=np.linspace(-0.02,0.045,10))
plt.xlabel('Rescaled value of the emission line flux')
plt.ylabel('Counts inside respective bin')
plt.title('Distribution of the rescaled line flux feature')
plt.savefig('./3a_flux.png')
plt.close()

#Put all of the features into an array so we can save it as a .txt file
features = np.array([kappa_scaled,color_scaled,extend_scaled,flux_scaled])
features = np.transpose(features)

#Save the rescaled features to a .txt file and display the first 10 lines
np.savetxt('features.txt',features,fmt='%1.8f')
first10lines = np.genfromtxt('features.txt')[0:10,:]
print('The first 10 lines of the features are\n',first10lines)
np.savetxt('handin4problem3.txt',first10lines,fmt='%1.8f')

print('The rest of the output file consists of the results from 3b and 3c:\n')

#PROBLEM 3B

def cost_function(xdata,ydata,weights,theta0):
    """
    The logistic cost function J(theta), takes input parameters xdata, which 
    are the m objects of the n features we use, so an m x n array. The ydata,
    which are the 0s and 1s corresponding to the m objects, n weights (theta),
    and lastly an arbitrary bias parameter theta0.
    """
    m = len(xdata) #m objects per feature
    z = np.matmul(weights,np.transpose(xdata)) #find the product x^T theta
    h = sigmoid(z+theta0) #call upon the sigmoid function including bias theta0
    J = ydata * np.log(h) + (1-ydata) * np.log(1-h) #compute the cost function
    return -np.sum(J)/m #the cost function is the sum of all J_i divided by m

def gradient_descent(xdata,ydata,weights,alpha,bias,iterations):
    """
    Inputs:
    xdata: the training data of the feature pair 
    ydata: the class of object i (spiral (1) or elliptical (0))
    weights: starting weights [1,1] to minimize
    alpha: learning rate, plays together with iterations, lower learning rate
    means the number of iterations needs to be higher. We use a constant value
    of 0.05 and vary the iterations instead. 
    bias: the bias = theta0 of the model. We just pick theta0 = 1.
    iterations: the number of iterations over which we minimize the cost func
    
    This is the gradient descent algorithm in which we use the gradient of 
    the cost function with respect to theta which we use to find the weights
    theta that minimize the cost function J. We append the cost function J
    at every step so we can track how the cost function decreases after each 
    iteration.
    """
    m = len(xdata) #m objects per feature 
    J_values = np.zeros(iterations) #array to store cost function 
    for j in range(iterations):
        z = np.matmul(weights,np.transpose(xdata)) #theta (weights) times x^T 
        h = sigmoid(z+bias) #hypothesis including bias 
        weights = weights - alpha * np.matmul(np.transpose(xdata), (h-ydata))/m
        J = cost_function(xdata,ydata,weights,bias) #theta0 is the bias! 
        J_values[j] = J
    return J_values, weights

training_ydata = ylabel[:700] #array is used for each pair, y_data same for all

#Try minimization for combination color (feature 1) and extend (feature 2)

features_12 = np.array([color_scaled,extend_scaled])
features_12 = np.transpose(features_12) #the data of features 1 and 2
training_xdata12 = features_12[:700] #use 70% of the data for training
weights_12 = np.ones(len(features_12[1])) #start with weights = 1 

min_J_12, min_weights_12 =\
    gradient_descent(training_xdata12,training_ydata,weights_12,0.05,1,2000)
print('The minimized weight values for color and extend are',min_weights_12)
                     
plt.plot(np.arange(0,2000,1),min_J_12,label=r'$\theta_1$ =')
plt.xlabel('Number of iterations N')
plt.ylabel(r'Cost function J($\theta$)')
plt.title(r'Convergence of J($\theta$) using features color and extend')
plt.savefig('./3b_cost12.png')
plt.close()


#Try minimization for combination kappa_co (feature 0) and flux (feature 3)

features_03 = np.array([kappa_scaled,flux_scaled])
features_03 = np.transpose(features_03)
training_xdata03 = features_03[:700]
weights_03 = np.ones(len(features_03[1]))

min_J_03, min_weights_03 =\
    gradient_descent(training_xdata03,training_ydata,weights_03,0.05,1,2000)
print('The minimized weight values for kappa_co and flux are',min_weights_03)
                     
plt.plot(np.arange(0,2000,1),min_J_03)
plt.xlabel('Number of iterations N')
plt.ylabel(r'Cost function J($\theta$)')
plt.title(r'Convergence of J($\theta$) using features $\kappa_{co}$ and flux')
plt.savefig('./3b_cost03.png')
plt.close()


#Try minimization for combination kappa_co (feature 0) and color (feature 1)

features_01 = np.array([kappa_scaled,color_scaled])
features_01 = np.transpose(features_01)
training_xdata01 = features_01[:700]
weights_01 = np.ones(len(features_01[1]))

min_J_01, min_weights_01 =\
    gradient_descent(training_xdata01,training_ydata,weights_01,0.05,1,3000)
print('The minimized weight values for kappa_co and color are',min_weights_01)
                     
plt.plot(np.arange(0,3000,1),min_J_01)
plt.xlabel('Number of iterations N')
plt.ylabel(r'Cost function J($\theta$)')
plt.title(r'Convergence of J($\theta$) using features $\kappa_{co}$ and color')
plt.savefig('./3b_cost01.png')
plt.close()

#Try minimization for combination extend (feature 2) and line-flux (feature 3)

features_23 = np.array([extend_scaled,flux_scaled])
features_23 = np.transpose(features_23)
training_xdata23 = features_23[:700]
weights_23 = np.ones(len(features_23[1]))

min_J_23, min_weights_23 =\
    gradient_descent(training_xdata23,training_ydata,weights_23,0.05,1,1000)
print('The minimized weight values for extend and lineflux are',min_weights_23)
                     
plt.plot(np.arange(0,1000,1),min_J_23)
plt.xlabel('Number of iterations N')
plt.ylabel(r'Cost function J($\theta$)')
plt.title(r'Convergence of J($\theta$) using features extend and flux')
plt.savefig('./3b_cost23.png')
plt.close()

#Try minimization for all features to see how the features rank 

features_all = np.array([kappa_scaled,color_scaled,extend_scaled,flux_scaled])
features_all = np.transpose(features_all)
training_xdata_all = features_all[:700] #use 70% of the data for training
weights_all = np.ones(len(features_all[1])) #start with weights = 1 

min_J_all, min_weights_all =\
   gradient_descent(training_xdata_all,training_ydata,weights_all,0.05,1,3000)
print('The minimized weight values using all features are',\
      min_weights_all)
                     
plt.plot(np.arange(0,3000,1),min_J_all,label=r'$\theta_1$ =')
plt.xlabel('Number of iterations N')
plt.ylabel(r'Cost function J($\theta$)')
plt.title(r'Convergence of J($\theta$) for all features')
plt.savefig('./3b_costall.png')
plt.close()


#PROBLEM 3C

def y_model(xdata,model_weights,bias):
    """
    Calculates the hypothesis h_theta(x^i) for object i. Takes input arrays 
    xdata, which is the data we are testing with our model, and model_weights
    which are the weights the model found that supposedly minimizes the cost
    function. We use np.where to catagorize class 0 and 1 (sigmoid function)
    """
    m = len(xdata) #m objects per feature 
    y_model = np.zeros(m)
    for i in range(m):
        z = np.matmul(model_weights,np.transpose(xdata[i]))
        y_model[i] = sigmoid(z+bias)
    y_model = np.where(y_model < 0.5, 0, 1) 
    return y_model

def confusion_matrix(ydata,y_model):
    """
    This function takes the y_data and predicted y as input to compare and see
    whether the model results in a true/false positive/negative. 
    Returns a 2x2 array
    """
    m = len(ydata) #number of objects
    CM_11, CM_12, CM_21, CM_22 = 0,0,0,0
    for i in range(m):
        if y_model[i] == 1 and ydata[i] == 1: #true positive
            CM_11 += 1
        elif y_model[i] - ydata[i] == 1: #false positive
            CM_12 += 1
        elif y_model[i] - ydata[i] == -1: #false negative
            CM_21 += 1 
        else: #true negative: ymodel_i and ydata_i are both 0
            CM_22 += 1
    return np.array([[CM_11,CM_12],[CM_21,CM_22]])

def F_score(beta,CM): #input a 2x2 confusion matrix and a beta
    """
    Uses the confusion matrix and a custom beta parameter to find the
    F_score as a function of beta. In this case we use beta = 1 everywhere,
    the final return is the equation used to find F_score(beta)
    """
    TP, FP, FN = CM[0,0],CM[0,1],CM[1,0] #get true pos, false pos and false neg
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    return (1 + beta**2) * (precision * recall)/((beta**2)*precision + recall)

def positives_negatives(feature_pair):
    """
    Splits the features into arrays distinguishing between class 0 and 1 using
    the data labels (spiral or elliptical column)
    """
    N = len(test_ylabel)
    pos = []
    neg = []
    for i in range(N):
        if test_ylabel[i] == 1:
            pos.append(feature_pair[i])
        else:
            neg.append(feature_pair[i])
    return np.array(pos),np.array(neg)
       
def decision_boundary(features,weights):
    """
    Compute the decision boundary using the following equation:
    y = -theta_0 * x / theta_1 - theta_0 / theta_2, using bias = theta0 = 1
    """
    return -(weights[0]*features[:,0]+1)/weights[1]

test_ylabel = ylabel[700:] #also the same for each of the feature pairs!

#For the 4 feature pairs we used in b we will now compute the F1 score and 
#scatter plot them, together with the decision boundary 

#try features color (1) and extend (2) as a combination


test_xdata12 = features_12[700:]
ymodel_12 = y_model(test_xdata12,min_weights_12,1)

CM_features12 = confusion_matrix(test_ylabel,ymodel_12)
print('The confusion matrix for color and extend is given by\n',CM_features12)
F1_score_12 = F_score(1,CM_features12)
print('The F1 score of the training set using color and extend is',F1_score_12)

decision_boundary_12 = decision_boundary(features_12,min_weights_12)
positives_12, negatives_12 = positives_negatives(test_xdata12)

plt.scatter(positives_12[:,0],positives_12[:,1],s=10,label='Spiral galaxy',\
            zorder=10)
plt.scatter(negatives_12[:,0],negatives_12[:,1],s=10,\
            label='Elliptical galaxy',zorder=2)
plt.plot(features_12[:,0],decision_boundary_12,color='green',\
         linestyle='dotted',linewidth=1,label='decision boundary')
plt.xlabel(r'Value of color')
plt.ylabel('Value of the galaxy extension')
plt.title(r'Color against extend, including decision boundary')
plt.xlim([-3,3])
plt.ylim([-0.5,1.22])
plt.legend(loc='upper left')
plt.savefig('./3c_plot12.png')
plt.close()

#try features kappa_co (0) and flux (3) as a combination

test_xdata03 = features_03[700:]
ymodel_03 = y_model(test_xdata03,min_weights_03,1)

CM_features03 = confusion_matrix(test_ylabel,ymodel_03)
print('The confusion matrix for kappa and flux is given by\n',\
      CM_features03)
F1_score_03 = F_score(1,CM_features03)
print('The F1 score of the training set using kappa and flux is',\
      F1_score_03)

decision_boundary_03 = decision_boundary(features_03,min_weights_03)
positives_03, negatives_03 = positives_negatives(test_xdata03)

plt.scatter(positives_03[:,0],positives_03[:,1],s=10,label='Spiral galaxy',\
            zorder=10)
plt.scatter(negatives_03[:,0],negatives_03[:,1],s=10,\
            label='Elliptical galaxy',zorder=12)
plt.plot(features_03[:,0],decision_boundary_03,color='green',\
         linestyle='dotted',linewidth=1,label='decision boundary')
plt.xlabel(r'Value of $\kappa_{co}$')
plt.ylabel('Value of the line-emission flux')
plt.title(r'$\kappa_{co}$ against flux, including decision boundary')
plt.xlim([-2,2])
plt.ylim([-0.07,0.07])
plt.legend(loc='upper left')
plt.savefig('./3c_plot03.png')
plt.close()

#try features kappa_co (0) and color (1) as a combination

test_xdata01 = features_01[700:]
ymodel_01 = y_model(test_xdata01,min_weights_01,1)

CM_features01 = confusion_matrix(test_ylabel,ymodel_01)
print('The confusion matrix for kappa and color is given by\n',CM_features01)
F1_score_01 = F_score(1,CM_features01)
print('The F1 score of the training set using kappa and color is',F1_score_01)

decision_boundary_01 = decision_boundary(features_01,min_weights_01)
positives_01, negatives_01 = positives_negatives(test_xdata01)

plt.scatter(positives_01[:,0],positives_01[:,1],s=10,label='Spiral galaxy',\
            zorder=10)
plt.scatter(negatives_01[:,0],negatives_01[:,1],s=10,\
            label='Elliptical galaxy',zorder=2)
plt.plot(features_01[:,0],decision_boundary_01,color='green',\
         linestyle='dotted',linewidth=1,label='decision boundary')
plt.xlabel(r'Value of $\kappa_{co}$')
plt.ylabel('Value of color')
plt.title(r'$\kappa_{co}$ against color, including decision boundary')
plt.xlim([-2,2])
plt.ylim([-3.2,3])
plt.legend(loc='upper left')
plt.savefig('./3c_plot01.png')
plt.close()

#try features extend (2) and line flux (3) as a combination

test_xdata23 = features_23[700:]
ymodel_23 = y_model(test_xdata23,min_weights_23,1)

CM_features23 = confusion_matrix(test_ylabel,ymodel_23)
print('The confusion matrix for extend and flux is given by\n',CM_features23)
F1_score_23 = F_score(1,CM_features23)
print('The F1 score of the training set using extend and flux is',F1_score_23)
    
decision_boundary_23 = decision_boundary(features_23,min_weights_23)
positives_23, negatives_23 = positives_negatives(test_xdata23)

plt.scatter(positives_23[:,0],positives_23[:,1],s=10,label='Spiral galaxy',\
            zorder=10)
plt.scatter(negatives_23[:,0],negatives_23[:,1],s=10,\
            label='Elliptical galaxy',zorder=2)
plt.plot(features_23[:,0],decision_boundary_23,color='green',\
         linestyle='dotted',linewidth=1,label='decision boundary')
plt.xlabel(r'Value of the galaxy extension')
plt.ylabel('Value of the line-emission flux')
plt.title(r'Extend against flux, including decision boundary')
plt.xlim([-0.4,3])
plt.ylim([-0.5,16])
plt.legend(loc='upper right')
plt.savefig('./3c_plot23.png')
plt.close()

#try all features

test_xdata_all = features_all[700:]
ymodel_all = y_model(test_xdata_all,min_weights_all,1)

CM_features_all = confusion_matrix(test_ylabel,ymodel_all)
print('The confusion matrix using all features is given by\n',\
      CM_features_all)
F1_score_all = F_score(1,CM_features_all)
print('The F1 score of the training set using all features is'\
      ,F1_score_all)

