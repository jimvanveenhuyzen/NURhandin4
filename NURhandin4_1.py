from astropy.time import Time
# import some coordinate things from astropy
from astropy.coordinates import solar_system_ephemeris
from astropy.coordinates import get_body_barycentric_posvel
from astropy import units as u

import numpy as np
import matplotlib.pyplot as plt

#PROBLEM 1A

#Current time from example
t = Time("2021-12-07 10:00")

#Initialize all the planets
with solar_system_ephemeris.set('jpl'):
    earth = get_body_barycentric_posvel('earth', t)
    sun = get_body_barycentric_posvel('sun', t)
    mercury = get_body_barycentric_posvel('mercury', t)
    venus = get_body_barycentric_posvel('venus', t)
    mars = get_body_barycentric_posvel('mars', t)
    jupiter = get_body_barycentric_posvel('jupiter', t)
    saturn = get_body_barycentric_posvel('saturn', t)
    uranus = get_body_barycentric_posvel('uranus', t)
    neptune = get_body_barycentric_posvel('neptune', t)

#calculate x,y,z positions & x,y,z velocities in AU and AU/day respectively
earthposition = earth[0]
earthvelocity = earth[1]
earthpos = [earthposition.x.to_value(u.AU),earthposition.y.to_value(u.AU),\
           earthposition.z.to_value(u.AU)]
earthvel = [earthvelocity.x.to_value(u.AU/u.d),earthvelocity\
           .y.to_value(u.AU/u.d), earthvelocity.z.to_value(u.AU/u.d)]
sunposition = sun[0]
sunvelocity = sun[1]
sunpos = [sunposition.x.to_value(u.AU),sunposition.y.to_value(u.AU),\
           sunposition.z.to_value(u.AU)]
sunvel = [sunvelocity.x.to_value(u.AU/u.d),sunvelocity\
           .y.to_value(u.AU/u.d), sunvelocity.z.to_value(u.AU/u.d)]
mercuryposition = mercury[0]
mercuryvelocity = mercury[1]
mercurypos = [mercuryposition.x.to_value(u.AU),mercuryposition\
              .y.to_value(u.AU),mercuryposition.z.to_value(u.AU)]
mercuryvel = [mercuryvelocity.x.to_value(u.AU/u.d),mercuryvelocity\
           .y.to_value(u.AU/u.d), mercuryvelocity.z.to_value(u.AU/u.d)]
venusposition = venus[0]
venusvelocity = venus[1]
venuspos = [venusposition.x.to_value(u.AU),venusposition\
              .y.to_value(u.AU),venusposition.z.to_value(u.AU)]
venusvel = [venusvelocity.x.to_value(u.AU/u.d),venusvelocity\
           .y.to_value(u.AU/u.d), venusvelocity.z.to_value(u.AU/u.d)]
marsposition = mars[0]
marsvelocity = mars[1]
marspos = [marsposition.x.to_value(u.AU),marsposition.y.to_value(u.AU),\
           marsposition.z.to_value(u.AU)]
marsvel = [marsvelocity.x.to_value(u.AU/u.d),marsvelocity\
           .y.to_value(u.AU/u.d), marsvelocity.z.to_value(u.AU/u.d)]
jupiterposition = jupiter[0]
jupitervelocity = jupiter[1]
jupiterpos = [jupiterposition.x.to_value(u.AU),jupiterposition.y.to_value\
              (u.AU),jupiterposition.z.to_value(u.AU)]
jupitervel = [jupitervelocity.x.to_value(u.AU/u.d),jupitervelocity\
           .y.to_value(u.AU/u.d), jupitervelocity.z.to_value(u.AU/u.d)]
saturnposition = saturn[0]
saturnvelocity = saturn[1]
saturnpos = [saturnposition.x.to_value(u.AU),saturnposition.y.to_value\
              (u.AU),saturnposition.z.to_value(u.AU)]
saturnvel = [saturnvelocity.x.to_value(u.AU/u.d),saturnvelocity\
           .y.to_value(u.AU/u.d), saturnvelocity.z.to_value(u.AU/u.d)]
uranusposition = uranus[0]
uranusvelocity = uranus[1]
uranuspos = [uranusposition.x.to_value(u.AU),uranusposition.y.to_value\
              (u.AU),uranusposition.z.to_value(u.AU)]
uranusvel = [uranusvelocity.x.to_value(u.AU/u.d),uranusvelocity\
           .y.to_value(u.AU/u.d), uranusvelocity.z.to_value(u.AU/u.d)]
neptuneposition = neptune[0]
neptunevelocity = neptune[1]
neptunepos = [neptuneposition.x.to_value(u.AU),neptuneposition.y.to_value\
              (u.AU),neptuneposition.z.to_value(u.AU)]
neptunevel = [neptunevelocity.x.to_value(u.AU/u.d),neptunevelocity\
           .y.to_value(u.AU/u.d), neptunevelocity.z.to_value(u.AU/u.d)]
    
#1a plots of the coordinates of the Sun and planets in the Solar system 
#This part plots the (x,y) position of the bodies at the current time
plt.scatter(earthpos[0],earthpos[1],label='Earth',color='royalblue'\
            ,s=15,zorder=10)
plt.scatter(sunpos[0],sunpos[1],label='Sun',color='darkorange',zorder=10)
plt.scatter(mercurypos[0],mercurypos[1],label='Mercury',s=15,zorder=10,\
            color='saddlebrown')
plt.scatter(venuspos[0],venuspos[1],label='Venus',s=15,zorder=10,\
            color='hotpink')
plt.scatter(marspos[0],marspos[1],label='Mars',color='red',s=15)
plt.scatter(jupiterpos[0],jupiterpos[1],label='Jupiter',s=15,color='peru')
plt.scatter(saturnpos[0],saturnpos[1],label='Saturn',s=15,color='gold')
plt.scatter(uranuspos[0],uranuspos[1],label='Uranus',s=15,color='lightblue')
plt.scatter(neptunepos[0],neptunepos[1],label='Neptune',s=15,color='darkblue')
plt.xlabel('x-position [AU]')
plt.ylabel('y-position [AU]')
plt.xlim([-5,32])
plt.ylim([-10,15])
plt.title('(x,y) positions of the Solar system at 2021-12-07 10:00')
plt.legend(loc='upper right')
plt.savefig('./1a_xyplot.png')
plt.close()

#This part plots the (x,z) position of the bodies at the current time
plt.scatter(earthpos[0],earthpos[2],label='Earth',color='royalblue',zorder=10,\
            s=15)
plt.scatter(sunpos[0],sunpos[2],label='Sun',s=40,color='darkorange',zorder=10)
plt.scatter(mercurypos[0],mercurypos[2],label='Mercury',zorder=10,\
            color='saddlebrown',s=15)
plt.scatter(venuspos[0],venuspos[2],label='Venus',\
            color='hotpink',s=15)
plt.scatter(marspos[0],marspos[2],label='Mars',color='red',s=15)
plt.scatter(jupiterpos[0],jupiterpos[2],label='Jupiter',color='peru',s=15)
plt.scatter(saturnpos[0],saturnpos[2],label='Saturn',s=15,color='gold')
plt.scatter(uranuspos[0],uranuspos[2],label='Uranus',s=15,color='lightblue')
plt.scatter(neptunepos[0],neptunepos[2],label='Neptune',s=15,color='darkblue')
plt.xlabel('x-position [AU]')
plt.ylabel('z-position [AU]')
plt.xlim([-5,32])
plt.ylim([-4,6])
plt.title('(x,z) positions of the Solar system at 2021-12-07 10:00')
plt.legend()
plt.savefig('./1a_xzplot.png')
plt.close()

#PROBLEM 1B

from astropy import constants as const #import G and Msun as constants

G = const.G.to_value()
M_sun = const.M_sun.to_value()
AU_to_m = 1.495978707e11 #I know there is probably a much cleaner way to do
days_to_s = 86400 #these conversions using astropy, but I found this easier
r_sun = np.array(sunpos) * AU_to_m

def magnitude_squared(r):
    """
    Three dimensional input vector of the form r = (x,y,z), computes magnitude-
    squared
    """
    return r[0]**2 + r[1]**2 + r[2]**2

def Fgrav(t,r,v):
    return G*M_sun*(r_sun-r)*(magnitude_squared(r_sun-r))**(-1.5)

def euler3D_2ndorder(fun,t0,r0,v0,tN,h):
    """
    Explicit Euler method for a function(t,r,v) like the acceleration a(r),
    adapted from regular 2nd order ODE Euler method to 3D version
    """
    t = np.arange(t0,tN,h) #create array containing all times
    r = np.zeros((len(t),3)) #create position array r(t)
    v = np.copy(r) #create velocity array v(t)
    r[0,:] = r0 #initial condition r(t0) = r0
    v[0,:] = v0 #initial condition v(t0) = v0
    for i in range(len(r)-1):
        v[i+1,:] = v[i,:] + h*fun(t[i],r[i,:],0)
        r[i+1,:] = r[i,:] + h*v[i,:] #times t0 + h to tN are filled using this
    return r,v #obtain the final position r(t) and velocity v(t)

def leapfrog(fun,t0,r0,v0,tN,h):
    """
    Takes input function of the form f(t,r,v), starting time t0, initial
    conditions for the position and velocity r0 and v0, ending time tN and
    lastly timestep h. Kickstarted using the explicit Euler method. 
    """
    t = np.arange(t0,tN,h) #create array containing all times
    r = np.zeros((len(t),3)) #create position array r(t)
    v = np.copy(r) #create velocity array v(t)
    r[0,:] = r0 #initial condition r(t0) = r0
    v[0,:] = euler3D_2ndorder(fun,t0,r0,v0,t0+h,0.5*h)[1][1]
    #line above uses initial condition v(t0) = v0 to take half a timestep using
    #the explicit Euler method
    for i in range(len(r)-1):
        v[i+1,:] = v[i,:] + h*fun(t[i],r[i,:],0)
        r[i+1,:] = r[i,:] + h*v[i+1,:] #times t0 + h to tN are filled in r(t)
    return t/timerange*200,r/AU_to_m,v/AU_to_m*days_to_s #converting units

timerange = 73050*86400 #200 years in seconds

def lf_problem1b(pos,vel):
    """
    I made this function so we only have to do the leapfrog once per planet,
    as it is slow to do so multiple times. 
    """
    pos = np.array(pos) * AU_to_m #unit conversion
    vel = np.array(vel) * AU_to_m / days_to_s
    return leapfrog(Fgrav,0,pos,vel,timerange,0.5*days_to_s) #calls leapfrog

#Only call the function once and save everything as a variable to save time
lf_earth = lf_problem1b(earthpos,earthvel)
lf_mercury = lf_problem1b(mercurypos,mercuryvel)
lf_venus = lf_problem1b(venuspos,venusvel)
lf_mars = lf_problem1b(marspos,marsvel)
lf_jupiter = lf_problem1b(jupiterpos,jupitervel)
lf_saturn = lf_problem1b(saturnpos,saturnvel)
lf_uranus = lf_problem1b(uranuspos,uranusvel)
lf_neptune = lf_problem1b(neptunepos,neptunevel)

#This part plots the (x,y) plot during a timespan of 200 years for problem 1b
plt.plot(lf_earth[1][:,0],lf_earth[1][:,1],label='Earth',color='royalblue',\
         linewidth=0.5,zorder=10)
plt.scatter(sunpos[0],sunpos[1],s=30,label='Sun',color='darkorange',zorder=0)
plt.plot(lf_mercury[1][:,0],lf_mercury[1][:,1],label='Mercury',\
         color='saddlebrown',linewidth=0.5,zorder=10)
plt.plot(lf_venus[1][:,0],lf_venus[1][:,1],label='Venus',color='hotpink',\
         linewidth=0.5,zorder=10)
plt.plot(lf_mars[1][:,0],lf_mars[1][:,1],label='Mars',color='red',\
         linewidth=0.5,zorder=10)
plt.plot(lf_jupiter[1][:,0],lf_jupiter[1][:,1],label='Jupiter',color='peru')
plt.plot(lf_saturn[1][:,0],lf_saturn[1][:,1],label='Saturn',color='gold')
plt.plot(lf_uranus[1][:,0],lf_uranus[1][:,1],label='Uranus',color='lightblue')
plt.plot(lf_neptune[1][:,0],lf_neptune[1][:,1],label='Neptune',\
         color='darkblue')
plt.xlabel('x-position [AU]')
plt.ylabel('y-position [AU]')
plt.title('Orbits of the planets in the (x,y) plane using Leapfrog')
plt.legend(loc='upper right')
plt.savefig('./1b_xyplot.png')
plt.close()

#This part plots the (t,z) plot during a timespan of 200 years for problem 1b
plt.plot(lf_earth[0],lf_earth[1][:,2],label='Earth',color='royalblue',\
         linewidth=0.5)
plt.plot(lf_mercury[0],lf_mercury[1][:,2],label='Mercury',color='saddlebrown'\
         ,linewidth=0.5)
plt.plot(lf_venus[0],lf_venus[1][:,2],label='Venus',color='hotpink',\
         linewidth=0.5)
plt.plot(lf_mars[0],lf_mars[1][:,2],label='Mars',color='red',linewidth=0.5)
plt.plot(lf_jupiter[0],lf_jupiter[1][:,2],label='Jupiter',color='peru')
plt.plot(lf_saturn[0],lf_saturn[1][:,2],label='Saturn',color='gold')
plt.plot(lf_uranus[0],lf_uranus[1][:,2],label='Uranus',color='lightblue')
plt.plot(lf_neptune[0],lf_neptune[1][:,2],label='Neptune',color='darkblue')
plt.xlabel('time [years]')
plt.ylabel('z-position [AU]')
plt.title('Time t against height z for the planets using Leapfrog')
plt.legend(loc='upper right')
plt.savefig('./1b_tzplot.png')
plt.close()

#PROBLEM 1C

def rk4_2ndorder(fun,t0,r0,v0,tN,h):
    """
    We treat the velocity v as just another variable that the acceleration a 
    depends on, so a = F(t,r,v). This way we dynamically update the k-values 
    used to calculate the trajectory r. 
    """
    t = np.arange(t0,tN,h) #create array containing all times
    r = np.zeros((len(t),3)) #create position array r(t)
    v = np.copy(r) #create velocity array v(t)
    r[0,:] = r0 #initial condition r(t0) = r0
    v[0,:] = v0 #initial condition v(t0) = v0
    for i in range(len(r)-1): 
        
        #In this loop we calculate 4 k-values (4th order) to have a weighted 
        #combination of slopes to calculate the position, increasing stability
        #compared to the Euler method
        
        k1_vel = h*v[i,:]
        k1_acc = h*fun(t[i],r[i,:],v[i,:])
        
        k2_vel = h*(v[i,:]+0.5*k1_acc) #dynamically update k-values with prev.
        k2_acc = h*fun(t[i]+0.5*h,r[i,:]+0.5*k1_vel,v[i,:]+0.5*k1_vel)
        
        k3_vel = h*(v[i,:]+0.5*k2_acc)
        k3_acc = h*fun(t[i]+0.5*h,r[i,:]+0.5*k2_vel,v[i,:]+0.5*k2_vel)
        
        k4_vel = h*(v[i,:]+k3_acc)
        k4_acc = h*fun(t[i]+h,r[i,:]+k3_vel,v[i,:]+k3_vel)
        
        v[i+1,:] = v[i,:] + k1_acc/6 + k2_acc/3 + k3_acc/3 + k4_acc/6
        #acc is the slope of velocity dv/dt, so use it to find v(t)
        r[i+1,:] = r[i,:] + k1_vel/6 + k2_vel/3 + k3_vel/3 + k4_vel/6
        #vel is the slope of position dr/dt, so use it to find r(t)
        
    return t/timerange*200,r/AU_to_m,v/AU_to_m*days_to_s

def rk4_problem1c(pos,vel):
    """
    I made this function so we only have to do the RK4 once per planet,
    as it is extremely slow to do so multiple times. 
    """
    pos = np.array(pos) * AU_to_m
    vel = np.array(vel) * AU_to_m / days_to_s
    return rk4_2ndorder(Fgrav,0,pos,vel,timerange,0.5*days_to_s)

#Only call the function once and save everything as a variable to save time
rk4_earth = rk4_problem1c(earthpos,earthvel)
rk4_mercury = rk4_problem1c(mercurypos,mercuryvel)
rk4_venus = rk4_problem1c(venuspos,venusvel)
rk4_mars = rk4_problem1c(marspos,marsvel)
rk4_jupiter = rk4_problem1c(jupiterpos,jupitervel)
rk4_saturn = rk4_problem1c(saturnpos,saturnvel)
rk4_uranus = rk4_problem1c(uranuspos,uranusvel)
rk4_neptune = rk4_problem1c(neptunepos,neptunevel)

#This part plots the (x,y) plot during a timespan of 200 years for problem 1c
plt.plot(rk4_earth[1][:,0],rk4_earth[1][:,1],label='Earth',color='blue',\
         linewidth=0.5,zorder=10)
plt.scatter(sunpos[0],sunpos[1],s=30,label='Sun',color='darkorange',zorder=0)
plt.plot(rk4_mercury[1][:,0],rk4_mercury[1][:,1],label='Mercury',\
         color='saddlebrown',linewidth=0.5,zorder=10)
plt.plot(rk4_venus[1][:,0],rk4_venus[1][:,1],label='Venus',color='hotpink',\
             linewidth=0.5,zorder=10)
plt.plot(rk4_mars[1][:,0],rk4_mars[1][:,1],label='Mars',color='red',\
             linewidth=0.5,zorder=10)
plt.plot(rk4_jupiter[1][:,0],rk4_jupiter[1][:,1],label='Jupiter',\
         color='peru')
plt.plot(rk4_saturn[1][:,0],rk4_saturn[1][:,1],label='Saturn',\
         color='gold')
plt.plot(rk4_uranus[1][:,0],rk4_uranus[1][:,1],label='Uranus',\
         color='lightblue')
plt.plot(rk4_neptune[1][:,0],rk4_neptune[1][:,1],label='Neptune',\
         color='darkblue')
plt.xlabel('x-position [AU]')
plt.ylabel('y-position [AU]')
plt.title('Orbits of planets in (x,y) plane using 4th order Runge-Kutta')
plt.legend(loc='upper right')
plt.savefig('./1c_xyplot.png')
plt.close()

#This part plots the (t,z) plot during a timespan of 200 years for problem 1c
plt.plot(rk4_earth[0],rk4_earth[1][:,2],label='Earth',color='royalblue',\
         linewidth=0.5)
plt.plot(rk4_mercury[0],rk4_mercury[1][:,2],label='Mercury',color='saddlebrown'\
         ,linewidth=0.5)
plt.plot(rk4_venus[0],rk4_venus[1][:,2],label='Venus',color='hotpink',\
         linewidth=0.5)
plt.plot(rk4_mars[0],rk4_mars[1][:,2],label='Mars',color='red',linewidth=0.5)
plt.plot(rk4_jupiter[0],rk4_jupiter[1][:,2],label='Jupiter',color='peru')
plt.plot(rk4_saturn[0],rk4_saturn[1][:,2],label='Saturn',color='gold')
plt.plot(rk4_uranus[0],rk4_uranus[1][:,2],label='Uranus',color='lightblue')
plt.plot(rk4_neptune[0],rk4_neptune[1][:,2],label='Neptune',color='darkblue')
plt.xlabel('time [years]')
plt.ylabel('z-position [AU]')
plt.title('Time t against height z for the planets using RK4')
plt.legend(loc='upper right')
plt.savefig('./1c_tzplot.png')
plt.close()

#Compute the difference in x-positions between the planets over time for 200 yr
earth_xdiff = (lf_earth[1][:,0] - rk4_earth[1][:,0])
mercury_xdiff = (lf_mercury[1][:,0] - rk4_mercury[1][:,0])
venus_xdiff = (lf_venus[1][:,0] - rk4_venus[1][:,0])
mars_xdiff = (lf_mars[1][:,0] - rk4_mars[1][:,0])
jupiter_xdiff = (lf_jupiter[1][:,0] - rk4_jupiter[1][:,0])
saturn_xdiff = (lf_saturn[1][:,0] - rk4_saturn[1][:,0])
uranus_xdiff = (lf_uranus[1][:,0] - rk4_uranus[1][:,0])
neptune_xdiff = (lf_neptune[1][:,0] - rk4_neptune[1][:,0])

#Plot these differences against time
plt.plot(lf_earth[0],earth_xdiff,label='Earth',color='royalblue',\
         linewidth=0.2)
plt.plot(lf_mercury[0],mercury_xdiff,label='Mercury',color='saddlebrown',\
         linewidth=0.1)
plt.plot(lf_venus[0],venus_xdiff,label='Venus',color='hotpink',\
         linewidth=0.2)
plt.plot(lf_mars[0],mars_xdiff,label='Mars',color='red',\
         linewidth=0.2)
plt.plot(lf_jupiter[0],jupiter_xdiff,label='Jupiter',color='peru',\
         linewidth=0.2)
plt.plot(lf_saturn[0],saturn_xdiff,label='Saturn',color='gold',\
         linewidth=0.2)
plt.plot(lf_uranus[0],uranus_xdiff,label='Uranus',color='lightblue',\
         linewidth=0.2)
plt.plot(lf_neptune[0],neptune_xdiff,label='Neptune',color='darkblue',\
         linewidth=0.2)
plt.xlabel('time [years]')
plt.ylabel('Diff in x-position between planets [AU]')
plt.title('Diff in x-pos vs. time for planets between Leapfrog and RK4')
plt.legend(loc='upper right')
plt.savefig('./1c_xdifftplot.png')
plt.close()




