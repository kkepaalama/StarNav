#/usr/bin/python

import numpy as np
import math as m

#astropy
from astropy.time import Time
from astropy.coordinates import SkyCoord
import astropy.units as u

#data
from ost_output_papakolea import s1
from main import cel2ecef_car, Rot, car2sph, sph2car


#ITRS(Cel) to ITRS(ECEF)
def cel2ecef(time, cel, radec, mode):
    time = [time]
    t = Time(time, format='iso', scale='utc')
    if mode == 'sph':
        crs = SkyCoord(ra = radec[0]*u.degree, dec = radec[1]*u.degree, obstime = t, frame = 'icrs', unit = 'deg')
        trs = crs.itrs
    if mode == 'car':
        crs = SkyCoord(x = cel[0], y = cel[1], z = cel[2], obstime = t, frame = 'icrs', representation_type = 'cartesian')
        trs = crs.itrs
    e = np.array([trs.x, trs.y, trs.z])
    return e


#ITRS(ECEF)(Geocentric Coordinate Sys) to Local/ENU(East, North, Up)(Topocentric Coordinate Sys)
def ecef2enu(lat, lon, position, vector):
    '''
    Transforms vector in the Earth-centered, Earth-fixed (ECEF) frame to East, North, Up (ENU) local frame
    lat: latitude from star_vector class, declination (DEC) component of <RA, DEC>
    lon: longitude from star_vector class, right ascention (RA) component of <RA, DEC>
    
    '''
    T_EL = np.array([[-m.sin(lon),               m.cos(lon),                 0    , position[0]],
                     [-m.cos(lon)*m.sin(lat),   -m.sin(lon)*m.sin(lat), m.cos(lat), position[1]],
                     [ m.cos(lon)*m.cos(lat),    m.sin(lon)*m.cos(lat), m.sin(lat), position[2]],
                     [        0,                           0,                0,        1   ]])
    v = np.array([vector[0], vector[1], vector[2], 1])
    l = np.dot(T_EL, v)
    l = np.delete(l, 3)
    return l

#Local/ENU(East, North, Up)(Topocentric Coordinate Sys) to Rover(Body)
def enu2body(tilt, local_vector): #where tilt is a 3x1 orientation vector gathered from IMU data and OST output
    x_rot = np.array([[1, 0, 0],
                      [0, (np.cos(-tilt[0])), -np.sin(-tilt[0])],
                      [0, np.sin(-tilt[0]), np.cos(-tilt[0])]])
    y_rot = np.array([[np.cos(-tilt[1]), 0, np.sin(-tilt[1])],
                      [0, 1, 0],
                      [-np.sin(-tilt[1]), 0, np.cos(-tilt[1])]])
    z_rot = np.array([[np.cos(-tilt[2]), -np.sin(-tilt[2]), 0],
                      [np.sin(-tilt[2]), np.cos(-tilt[2]), 0],
                      [0, 0, 1]])
    R_LB = np.dot(x_rot, np.dot(y_rot, z_rot))
    b = np.dot(R_LB, local_vector)
    return b

#constannts
rad = np.pi/180

#ideal conditions
'''lat, lon = 21.295084*rad, -157.811978*rad
s_e = sph2car(lat, lon)
org_e = [0, 0, 0]
org_l = [s_e[0], s_e[1], s_e[2]]
org_b = org_l
init_e = s_e
init_l = ecef2enu(lat, lon, s_e, s_e)


tilt = [0*rad, 0.1*rad, 0.1*rad]    
init_b = enu2body(tilt, init_l)
R_BE = Rot(init_b, init_e)
Rdotp = np.dot(R_BE, init_e)
p_E = Rdotp + init_l
p_e = p_E/np.linalg.norm(p_E)
pe = car2sph(p_e)
print(pe)'''

'''for i in np.linspace(0, np.pi/2, num=11):
    tilt = [0, i, 0]    
    init_b = enu2body(tilt, init_l)
    R_BE = Rot(init_b, init_e)
    Rdotp = np.dot(R_BE, init_e)
    p_E = Rdotp + init_l
    p_e = p_E/np.linalg.norm(p_E)
    pe = car2sph(p_e)
    print(pe)'''


#origins-points
radec = [327.576163*rad, 18.175975*rad]
#org_l = cel2ecef(s1.time, s1.cel, radec, 'sph')
#org_l = [-0.886493, -0.340148, 0.313734]
org_l = [-0.512656, -0.858559, 0.00775253]

org_b = org_l

#vectors
#init_c = np.array([0, 0, 0]) #initial boresight coordinate in celestial frame in <RA, DEC>
#init_e = cel2ecef(s1.time, s1.cel, s1.radec, 'sph') #vector from org_e to star (since the vector is normalize it is also the point that intersects shpere with r = 1 in <x, y, z>)
init_e = np.array([-0.886493, -0.340148, 0.313734]) #shortcut to optimiz time (uncomment line above for full flexibility)
init_l = ecef2enu(s1.radec[1], s1.radec[0], org_l, init_e)
tilt = [0, 0, 0*rad]
#tilt = [0.06161487984311333, 0.03387764611236473, 23.6*rad] #pitch, roll, and yaw --> counter rotation of body frame to align with local frame
init_b = enu2body(tilt, init_l)
#init_b = init_b*10

R_BE = Rot(init_b, init_e) #rotation matrix from B to E
Rdotp = np.dot(R_BE, init_e) 
p_E = Rdotp + init_l
#p_E = p_E*10**-1
p_e = car2sph(p_E)
print(p_e)
