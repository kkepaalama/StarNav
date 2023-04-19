#/usr/bin/python

import numpy as np
import math as m

from astropy.time import Time
from astropy.coordinates import SkyCoord
import astropy.units as u

from main import car2sph, sph2car
from starsim_def import Rot

'''
assuptions:
    Earth is a perfect sphere, surface is smooth
    Rover sits flat on surface
    Gravity throughout Earth is uniform, points to Earth’s center
    L frame and B frame share the same origin
'''

#constants
rad = np.pi/180
r_earth = 6378000
    

v_EEC = [21*rad, 203*rad] #vector from the E to C frame expressed in E
v_EEC = sph2car(v_EEC[0], v_EEC[1]) #conversion of <RA,DEC> to cartesian

#L frame
origin_L = v_EEC #origin (point) of the L frame
n_LL = np.subtract(v_EEC,v_EEC*1.1) #normal vector of the L frame
g_LLE = n_LL*-1 #gravity vector from L to E expressed in L


#B frame
origin_B = origin_L
n_BB = np.subtract(v_EEC,v_EEC*1.1) #normal vector in the B frame
g_BBL = g_LLE #gravity vector from B to L expressed in B
angle_x = (90, 0, 0)
angle_y = (0, 90, 0)
Rx = Rot(angle_x)
Ry = Rot(angle_y)
x_B = np.dot(Rx, n_BB) #x-axis for B frame
y_B = np.dot(Ry, n_BB) #y-axis for B frame
z_B = n_BB #z-axis for B frame. Z-axis and normal vector are the same vector