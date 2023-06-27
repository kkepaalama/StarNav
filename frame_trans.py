#/usr/bin/python

import numpy as np
import math
import geopy.distance
import pymap3d

#astropy
from astropy.time import Time
from astropy.coordinates import SkyCoord
import astropy.units as u

#data
from ost_output_papakolea import s1
import main


#ITRS(Cel) to ITRS(ECEF)
def cel2ecef(time, cel, radec, mode):
    time = [time]
    t = Time(time, format='iso', scale='utc')
    if mode == 'radec2car':
        crs = SkyCoord(ra = radec[0]*u.degree, dec = radec[1]*u.degree, obstime = t, frame = 'icrs', unit = 'deg')
        trs = crs.itrs
        e = np.array([trs.x, trs.y, trs.z])
        return e
    if mode == 'radec2sph':
        crs = SkyCoord(ra = radec[0]*u.degree, dec = radec[1]*u.degree, obstime = t, frame = 'icrs', unit = 'deg')
        trs = crs.itrs
        trs.representation_type = 'spherical'
        return trs
    if mode == 'car':
        crs = SkyCoord(x = cel[0], y = cel[1], z = cel[2], obstime = t, frame = 'icrs', representation_type = 'cartesian')
        trs = crs.itrs
        e = np.array([trs.x, trs.y, trs.z])
        return e


#ITRS(ECEF)(Geocentric Coordinate Sys) to Local/ENU(East, North, Up)(Topocentric Coordinate Sys)
def ecef2enu(latitude, longitude, position, vector):
    '''
    Transforms vector in the Earth-centered, Earth-fixed (ECEF) frame to East, North, Up (ENU) local frame
    lat: latitude from star_vector class, declination (DEC) component of <RA, DEC>
    lon: longitude from star_vector class, right ascention (RA) component of <RA, DEC>.
       
    '''
    lat = np.radians(latitude)
    lon = np.radians(longitude)
    
    T_EL = np.array([[-math.sin(lon),                  math.cos(lon),                     0,       position[0]],
                     [-math.cos(lon)*math.sin(lat),   -math.sin(lon)*math.sin(lat), math.cos(lat), position[1]],
                     [ math.cos(lon)*math.cos(lat),    math.sin(lon)*math.cos(lat), math.sin(lat), position[2]],
                     [        0,                           0,                             0,           1     ]])
    v = np.array([vector[0], vector[1], vector[2], 1])
    l = np.dot(T_EL, v)
    l = np.delete(l, 3)
    return l


def ecef_to_enu(ecef_vector, latitude, longitude):
    # Convert latitude and longitude to radians
    phi = np.radians(latitude)
    lam = np.radians(longitude)

    # Calculate sine and cosine values
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    sin_lam = np.sin(lam)
    cos_lam = np.cos(lam)

    # Calculate transformation matrix
    transformation_matrix = np.array([[-sin_lam, cos_lam, 0],
                                      [-sin_phi * cos_lam, -sin_phi * sin_lam, cos_phi],
                                      [cos_phi * cos_lam, cos_phi * sin_lam, sin_phi]])

    # Perform the transformation
    enu_vector = np.dot(transformation_matrix, ecef_vector)
    return enu_vector


def enu_to_ecef(enu_vector, latitude, longitude):
    # Convert latitude and longitude to radians
    phi = np.radians(latitude)
    lam = np.radians(longitude)

    # Calculate sine and cosine values
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    sin_lam = np.sin(lam)
    cos_lam = np.cos(lam)

    # Calculate transformation matrix
    transformation_matrix = np.array([[-sin_lam, cos_lam, 0],
                                      [-sin_phi * cos_lam, -sin_phi * sin_lam, cos_phi],
                                      [cos_phi * cos_lam, cos_phi * sin_lam, sin_phi]])
    transpose = np.transpose(transformation_matrix)

    # Perform the transformation
    ecef_vector = np.dot(transpose, enu_vector)
    return ecef_vector


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
    R_LB = np.dot(z_rot, np.dot(y_rot, x_rot))
    b = np.dot(R_LB, local_vector)
    return b

def R(angles, local_vector):
    x_rot = np.array([[1, 0, 0],
                      [0, (np.cos(angles[0])), -np.sin(angles[0])],
                      [0, np.sin(angles[0]), np.cos(angles[0])]])
    y_rot = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                      [0, 1, 0],
                      [-np.sin(angles[1]), 0, np.cos(angles[1])]])
    z_rot = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                      [np.sin(angles[2]), np.cos(angles[2]), 0],
                      [0, 0, 1]])
    Rotation = np.dot(x_rot, np.dot(y_rot, z_rot))
    n = np.dot(Rotation, local_vector)
    return n

def R_AC(latitude, longitude, tilt):
    phi = np.radians(latitude)
    lam = np.radians(longitude)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    sin_lam = np.sin(lam)
    cos_lam = np.cos(lam)
    R_AB = np.array([[-sin_lam, cos_lam, 0],
                     [-sin_phi * cos_lam, -sin_phi * sin_lam, cos_phi],
                     [cos_phi * cos_lam, cos_phi * sin_lam, sin_phi]])
    Rx = np.array([[1, 0, 0],
                      [0, (np.cos(tilt[0])), -np.sin(tilt[0])],
                      [0, np.sin(tilt[0]), np.cos(tilt[0])]])
    Ry = np.array([[np.cos(tilt[1]), 0, np.sin(tilt[1])],
                      [0, 1, 0],
                      [-np.sin(tilt[1]), 0, np.cos(tilt[1])]])
    Rz = np.array([[np.cos(tilt[2]), -np.sin(tilt[2]), 0],
                      [np.sin(tilt[2]), np.cos(tilt[2]), 0],
                      [0, 0, 1]])
    R_BC = np.dot(Rx, np.dot(Ry, Rz))
    R_AC = np.dot(R_AB, R_BC)
    
    return R_AC


#constannts
gts = 21.295084, -157.811978 # <lat, lon> of ground truth in decimal degrees
gtc = np.array([-0.86272793, -0.35186237,  0.36317129])


#ideal conditions
'''lat, lon = 21.295084*rad, -157.811978*rad    #initial position
#lat, lon = 21*rad, -157*rad        #initial position
s_e = sph2car(lat, lon)         #conversion from spherical to cartesian
org_e = [0, 0, 0]       #origin of E frame in terms of E
org_l = [s_e[0], s_e[1], s_e[2]]        #origin of L frame
org_b = org_l       #origin of L and B frame coincide
init_e = s_e        #initial position in cartesian
init_l = ecef2enu(lat, lon, s_e, s_e)       #initial position for L frame


tilt = [0.1*rad, 0.0*rad, 0*rad]      #tilt
init_b = enu2body(tilt, init_l)         #initial position of B frame origin
R_BE = Rot(init_b, init_e)      #homogeneous transformation
Rdotp = np.dot(R_BE, init_e)
p_E = Rdotp + init_l
p_e = p_E/np.linalg.norm(p_E)
pe = car2sph(p_e)       #final position estimate
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
#radec = [323.113683*rad, 0.810916*rad]
#latlone = cel2ecef(s1.time, s1.cel, s1.radec, 'radec2sph')
#latlon_e = 18.28441551*rad, 200.99184445*rad #latlon_e shortcut (transfromed vector from cel to ecef)
#test = np.array([np.radians(18.28441551), np.radians(200.99184445)]) #21.295084*rad, -157.811978*rad
#est = [18.28441551, 200.99184445]
#car_e = cel2ecef(s1.time, s1.cel, s1.radec, 'radec2car') #transformed vector from cel to ecef (cartesian)
#s_e = sph2car(latlon_e[0], latlon_e[1]) 
#s_e = sph2car(test[0], test[1]) 
#org_l = np.array([s_e[0], s_e[1], s_e[2]])
#init_e = org_l

#vectors
#init_c = np.array([0, 0, 0]) #initial boresight coordinate in celestial frame in <RA, DEC>
#init_e = cel2ecef(s1.time, s1.cel, s1.radec, 'sph') #vector from org_e to star (since the vector is normalize it is also the point that intersects shpere with r = 1 in <x, y, z>)'''
#init_l = ecef2enu(latlon_e[0], latlon_e[1], org_l, org_l)
#s1.ecef = cel2ecef(s1.time, s1.cel[0], s1.radec, 'car')
#s1.ecef = np.array([-0.89138523, -0.39121153,  0.22887966])
#s1.ecef = [s1.ecef[0], s1.ecef[1], s1.ecef[2]]
#init_l = ecef_to_enu(s1.body[0], test[0], test[1])
#init_l = ecef_to_enu(s_e, 18.28441551, 200.99184445)
#tilt = np.array([0, 0, 0])
#tilt = [0.06161487984311333, 0.03387764611236473, -23.6*rad]
#init_b = R(tilt, init_l)
#init_b = enu2body(tilt, init_l)


BS1 = s1.body[0]
#tilt = [0.06161487984311333, 0.03387764611236473, -0.4118977034706618]
tilt = [-0.4118977034706618, 0.06161487984311333, 0.03387764611236473]
#tilt = [0, 0, 0]
RBL = main.R_inverse(tilt)
LS1 = np.dot(RBL, BS1)

ES1 = np.array([-0.89138523, -0.39121153,  0.22887966])*2

EL1 = ES1 - LS1


#R_BE = Rot(init_b, init_e) #rotation matrix from B to E
#Rdotp = np.dot(R_BE, init_e)
#p_E = (Rdotp + init_l)
#p_e = p_E/np.linalg.norm(p_E)
pe = main.car2sph(EL1)
print(pe)


coords_1 = (gts[0], gts[1])
coords_2 = (pe[0], pe[1])
print(geopy.distance.geodesic(coords_1, coords_2).miles, 'miles    ', geopy.distance.geodesic(coords_1, coords_2).km, 'km')

