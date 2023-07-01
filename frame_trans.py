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
def body2enu(tilt, body_vector): #where tilt is a 3x1 orientation vector gathered from IMU data and OST output
    x_rot = np.array([[1, 0, 0],
                      [0, (np.cos(-tilt[0])), -np.sin(-tilt[0])],
                      [0, np.sin(-tilt[0]), np.cos(-tilt[0])]])
    y_rot = np.array([[np.cos(-tilt[1]), 0, np.sin(-tilt[1])],
                      [0, 1, 0],
                      [-np.sin(-tilt[1]), 0, np.cos(-tilt[1])]])
    z_rot = np.array([[np.cos(-tilt[2]), -np.sin(-tilt[2]), 0],
                      [np.sin(-tilt[2]), np.cos(-tilt[2]), 0],
                      [0, 0, 1]])
    R_BL = np.dot(z_rot, np.dot(y_rot, x_rot))
    b = np.dot(R_BL, body_vector)
    return b

def rotation_matrix(angles, local_vector):
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


#constannts
gts = 21.295084, -157.811978 # <lat, lon> of ground truth in decimal degrees
gtc = np.array([-0.86272793, -0.35186237,  0.36317129])

E_EoS0 = cel2ecef(s1.time, s1.cel[0], s1.radec, 'car')
E_EoS1 = cel2ecef(s1.time, s1.cel[1], s1.radec, 'car')
E_EoS2 = cel2ecef(s1.time, s1.cel[2], s1.radec, 'car')
E_EoS3 = cel2ecef(s1.time, s1.cel[3], s1.radec, 'car')
E_EoS4 = cel2ecef(s1.time, s1.cel[4], s1.radec, 'car')
E_EoS5 = cel2ecef(s1.time, s1.cel[5], s1.radec, 'car')
E_EoS6 = cel2ecef(s1.time, s1.cel[6], s1.radec, 'car')
E_EoS7 = cel2ecef(s1.time, s1.cel[7], s1.radec, 'car')
E_EoS8 = cel2ecef(s1.time, s1.cel[8], s1.radec, 'car')
E_EoS9 = cel2ecef(s1.time, s1.cel[9], s1.radec, 'car')

tilt = [-0.4118977034706618, 0.06161487984311333, 0.03387764611236473]

B_BoS0 = body2enu(tilt, s1.body[0])
B_BoS1 = body2enu(tilt, s1.body[1])
B_BoS2 = body2enu(tilt, s1.body[2])
B_BoS3 = body2enu(tilt, s1.body[3])
B_BoS4 = body2enu(tilt, s1.body[4])
B_BoS5 = body2enu(tilt, s1.body[5])
B_BoS6 = body2enu(tilt, s1.body[6])
B_BoS7 = body2enu(tilt, s1.body[7])
B_BoS8 = body2enu(tilt, s1.body[8])
B_BoS9 = body2enu(tilt, s1.body[9])

B0 = main.B(B_BoS0, E_EoS0, s1.wt[0])
B1 = main.B(B_BoS1, E_EoS1, s1.wt[1])
B2 = main.B(B_BoS2, E_EoS2, s1.wt[2])
B3 = main.B(B_BoS3, E_EoS3, s1.wt[3])
B4 = main.B(B_BoS4, E_EoS4, s1.wt[4])
B5 = main.B(B_BoS5, E_EoS5, s1.wt[5])
B6 = main.B(B_BoS6, E_EoS6, s1.wt[6])
B7 = main.B(B_BoS7, E_EoS7, s1.wt[7])
B8 = main.B(B_BoS8, E_EoS8, s1.wt[8])
B9 = main.B(B_BoS9, E_EoS9, s1.wt[9])

B = B1 + B2 + B3 + B4 + B5 + B6 + B7 + B8 + B9
K = main.K(B)
q = main.q(K)
E_R_B = main.q2R(q)

E_p_B = E_EoS4 - np.dot(E_R_B, B_BoS4).reshape(3, 1)

global_position = main.car2sph(E_p_B)
print(global_position)


coords_1 = (gts[0], gts[1])
coords_2 = (global_position[0], global_position[1])
print(geopy.distance.geodesic(coords_1, coords_2).miles, 'miles    ', geopy.distance.geodesic(coords_1, coords_2).km, 'km')
