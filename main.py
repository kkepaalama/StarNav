#/usr/bin/python

### This file contains all the required functions for the global position estimate algorithm

import numpy as np
import math as m

from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy.coordinates import ITRS
from astropy.coordinates import ICRS
from astropy.coordinates import GCRS
import astropy.units as u


#ITRS(eci) to ITRS(ECEF)
def eci2ecef(time, eci, radec, mode):
    time = time
    t = Time(time, format='iso', scale='utc')
    if mode == 'radec2car':
        crs = SkyCoord(ra = radec[0]*u.degree, dec = radec[1]*u.degree, obstime = t, frame = 'gcrs', unit = 'deg')
        trs = crs.itrs
        radec2car = np.array([trs.x, trs.y, trs.z])
        return radec2car
    if mode == 'radec2sph':
        crs = SkyCoord(ra = radec[0]*u.degree, dec = radec[1]*u.degree, obstime = t, frame = 'icrs', unit = 'deg')
        trs = crs.itrs
        trs.representation_type = 'spherical'
        return trs
    if mode == 'car':
        crs = SkyCoord(x = eci[0], y = eci[1], z = eci[2], obstime = t, frame = 'icrs', representation_type = 'cartesian')
        trs = crs.itrs
        car = np.array([trs.x, trs.y, trs.z])
        return car
    if mode == 'sph':
        crs = SkyCoord(x = eci[0], y = eci[1], z = eci[2], obstime = t, frame = 'icrs', representation_type = 'cartesian')
        trs = crs.itrs
        trs.representation_type = 'spherical'
        return trs
    if mode == 'gcrs':
        gcrs = SkyCoord(x = eci[0], y = eci[1], z = eci[2], obstime = t, frame = 'gcrs', representation_type = 'cartesian')
        trs = gcrs.itrs
        trs = np.array([trs.x, trs.y, trs.z])
        return trs

###Davenport q-Method
#B matrix
def B(body, trs, w):
    body = body.reshape(3, 1)
    trs = trs.reshape(3, 1)
    B = np.dot(w, np.dot(body, np.transpose(trs)))
    return B

#K matrix
def K(B):
    S = B + np.transpose(B)
    Z = np.array([[B[1, 2] - B[2, 1]], 
                  [B[2, 0] - B[0, 2]],
                  [B[0, 1] - B[1, 0]]])
    sigma = np.array([[B.trace()]])
    top_right = S - sigma*np.identity(3)
    K1 = np.concatenate((top_right, np.transpose(Z)), axis = 0)
    K2 = np.concatenate((Z, sigma), axis = 0)
    K = np.concatenate((K1, K2), axis = 1)
    return K
    
def q(K):
    eigval, eigvec = np.linalg.eig(K)
    if np.max(eigval) == eigval[0]:
        return eigvec[:, 0]
    elif np.max(eigval) == eigval[1]:
        return eigvec[:, 1]
    elif np.max(eigval) == eigval[2]:
        return eigvec[:, 2] 
    elif np.max(eigval) == eigval[3]:
        return eigvec[:, 3]

def n(q):
    n1 = q[0]/(np.sin(np.arccos(q[3])))
    n2 = q[1]/(np.sin(np.arccos(q[3])))
    n3 = q[2]/(np.sin(np.arccos(q[3])))
    return np.array([[n1], [n2], [n3]])


def imu2cam(imu_tilt):
    T = np.array([[-0.0184015, 0.82131394, 0.57017962, -0.07331063],
                  [ 0.37333988, -0.5233833, 0.76595513, -0.05248031],
                  [ 0.92751211, 0.22696551, -0.29699821, -0.00126057],
                  [ 0, 0, 0, 1 ]])
    cam_tilt = T@imu_tilt
    cam_tilt = np.delete(cam_tilt, 3)
    return cam_tilt


def pos(cam_tilt, n):
    x_rot = np.array([[1, 0, 0],
                      [0, (np.cos(-cam_tilt[0])), -np.sin(-cam_tilt[0])],
                      [0, np.sin(-cam_tilt[0]), np.cos(-cam_tilt[0])]])
    y_rot = np.array([[np.cos(-cam_tilt[1]), 0, np.sin(-cam_tilt[1])],
                      [0, 1, 0],
                      [-np.sin(-cam_tilt[1]), 0, np.cos(-cam_tilt[1])]])
    z_rot = np.array([[np.cos(-cam_tilt[2]), -np.sin(-cam_tilt[2]), 0],
                      [np.sin(-cam_tilt[2]), np.cos(-cam_tilt[2]), 0],
                      [0, 0, 1]])
    R = np.dot(z_rot, np.dot(y_rot, x_rot))
    n_t = np.dot(R, n)
    lat = m.degrees(np.pi/2) - m.degrees(np.arccos(n_t[2]))
    lon = m.degrees(np.arctan2(n_t[1], n_t[0]))
    if (lon < 0):
        lon = lon + 360
    return lat, lon

'''def R(tilt):
    x_rot = np.array([[1, 0, 0],
                      [0, (np.cos(tilt[0])), -np.sin(tilt[0])],
                      [0, np.sin(tilt[0]), np.cos(tilt[0])]])
    y_rot = np.array([[np.cos(tilt[1]), 0, np.sin(tilt[1])],
                      [0, 1, 0],
                      [-np.sin(tilt[1]), 0, np.cos(tilt[1])]])
    z_rot = np.array([[np.cos(tilt[2]), -np.sin(tilt[2]), 0],
                      [np.sin(tilt[2]), np.cos(tilt[2]), 0],
                      [0, 0, 1]])
    R = np.dot(z_rot, np.dot(y_rot, x_rot))
    return R'''

def Rotation_XYZ(tilt):
    x_rot = np.array([[1, 0, 0],
                      [0, (np.cos(tilt[0])), -np.sin(tilt[0])],
                      [0, np.sin(tilt[0]), np.cos(tilt[0])]])
    y_rot = np.array([[np.cos(tilt[1]), 0, np.sin(tilt[1])],
                      [0, 1, 0],
                      [-np.sin(tilt[1]), 0, np.cos(tilt[1])]])
    z_rot = np.array([[np.cos(tilt[2]), -np.sin(tilt[2]), 0],
                      [np.sin(tilt[2]), np.cos(tilt[2]), 0],
                      [0, 0, 1]])
    n = np.dot(x_rot, np.dot(y_rot, z_rot))
    return n


def Rotation_ZYX(tilt):    #this is the correct euler counter rotation DO NOT CHANGE
    x_rot = np.array([[1, 0, 0],
                      [0, (np.cos(-tilt[0])), -np.sin(-tilt[0])],
                      [0, np.sin(-tilt[0]), np.cos(-tilt[0])]])
    y_rot = np.array([[np.cos(-tilt[1]), 0, np.sin(-tilt[1])],
                      [0, 1, 0],
                      [-np.sin(-tilt[1]), 0, np.cos(-tilt[1])]])
    z_rot = np.array([[np.cos(-tilt[2]), -np.sin(-tilt[2]), 0],
                      [np.sin(-tilt[2]), np.cos(-tilt[2]), 0],
                      [0, 0, 1]])
    Rotation_ZYX = np.dot(z_rot, np.dot(y_rot, x_rot))
    return Rotation_ZYX

def rotation_matrix(tilt, local_vector):
    x_rot = np.array([[1, 0, 0],
                      [0, (np.cos(tilt[0])), -np.sin(tilt[0])],
                      [0, np.sin(tilt[0]), np.cos(tilt[0])]])
    y_rot = np.array([[np.cos(tilt[1]), 0, np.sin(tilt[1])],
                      [0, 1, 0],
                      [-np.sin(tilt[1]), 0, np.cos(tilt[1])]])
    z_rot = np.array([[np.cos(tilt[2]), -np.sin(tilt[2]), 0],
                      [np.sin(tilt[2]), np.cos(tilt[2]), 0],
                      [0, 0, 1]])
    Rotation = np.dot(x_rot, np.dot(y_rot, z_rot))
    n = np.dot(Rotation, local_vector)
    return n

def Rot(a, b):
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    if s == 0:
       return print('vectors are parallel')
    else:
        c = np.dot(a, b)
        skew = np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])
        R = np.eye(3) + skew + skew.dot(skew) * ((1 - c)/(s ** 2 ))
    return R

def sph2car(lat, lon):
    x = np.cos(lat)*np.cos(lon)
    y = np.cos(lat)*np.sin(lon)
    z = np.sin(lat)
    vec = np.array([x, y, z])
    return vec

def car2sph(n):
    lat = m.degrees(np.pi/2) - m.degrees(np.arccos(n[2]))
    lon = m.degrees(np.arctan2(n[1], n[0]))
    return (lat, lon)

def q2R(q):
    # Extract the values from q
    # where q = (q0, q1, q2, q3) = (q0, q123) where q123 = <qi, qj, qk>
    q0 = q[3]
    q1 = q[0]
    q2 = q[1]
    q3 = q[2]
     
    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
     
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
     
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
     
    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])
                            
    return rot_matrix

#ITRS(ECEF)(Geocentric Coordinate Sys) to Local/ENU(East, North, Up)(Topocentric Coordinate Sys)
def ecef2enu(latitude, longitude, position, vector):
    '''
    Transforms vector in the Earth-centered, Earth-fixed (ECEF) frame to East, North, Up (ENU) local frame
    lat: latitude from star_vector class, declination (DEC) component of <RA, DEC>
    lon: longitude from star_vector class, right ascention (RA) component of <RA, DEC>.
       
    '''
    lat = np.radians(latitude)
    lon = np.radians(longitude)
    
    T_EL = np.array([[-m.sin(lon),                  m.cos(lon),                     0,       position[0]],
                     [-m.cos(lon)*m.sin(lat),   -m.sin(lon)*m.sin(lat), m.cos(lat), position[1]],
                     [ m.cos(lon)*m.cos(lat),    m.sin(lon)*m.cos(lat), m.sin(lat), position[2]],
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
def body2enu(tilt, body_vector): #where tilt is a 3x1 tilt vector gathered from IMU data and OST output
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


def lla2ecef(latitude, longitude, altitude):
    # Constants for WGS84 ellipsoid
    a = 6378137.0  # Semi-major axis
    b = 6356752.314245  # Semi-minor axis
    e_sq = (a**2 - b**2) / a**2  # Eccentricity squared

    # Convert latitude and longitude to radians
    lat_rad = m.radians(latitude)
    lon_rad = m.radians(longitude)

    # Calculate radius of curvature in the prime vertical
    N = a / m.sqrt(1 - e_sq * m.sin(lat_rad)**2)

    # Calculate ECEF coordinates
    x = (N + altitude) * m.cos(lat_rad) * m.cos(lon_rad)
    y = (N + altitude) * m.cos(lat_rad) * m.sin(lon_rad)
    z = ((b**2 / a**2) * N + altitude) * m.sin(lat_rad)

    return x, y, z

def Rx(theta):
    rx = np.array([[1, 0, 0],
                   [0, (np.cos(theta)), -np.sin(theta)],
                   [0, np.sin(theta), np.cos(theta)]])
    return rx
    
def Ry(theta):
    ry = np.array([[np.cos(theta), 0, np.sin(theta)],
                   [0, 1, 0],
                   [-np.sin(theta), 0, np.cos(theta)]])
    return ry

def Rz(theta):
    rz = np.array([[np.cos(theta), -np.sin(theta), 0],
                   [np.sin(theta), np.cos(theta), 0],
                   [0, 0, 1]])
    return rz

def euler(phi, theta, psi, mode):
    rx = np.array([[1, 0, 0],
                   [0, (np.cos(phi)), -np.sin(phi)],
                   [0, np.sin(phi), np.cos(phi)]])
    ry = np.array([[np.cos(theta), 0, np.sin(theta)],
                   [0, 1, 0],
                   [-np.sin(theta), 0, np.cos(theta)]])
    rz = np.array([[np.cos(psi), -np.sin(psi), 0],
                   [np.sin(psi), np.cos(psi), 0],
                   [0, 0, 1]])
    if mode == 'XYZ':
        xyz = np.dot(rz, np.dot(ry, rx))
        return xyz
    if mode == 'XZY':
        xzy = np.dot(ry, np.dot(rz, rx))
        return xzy
    if mode == 'YZX':
        yzx = np.dot(rx, np.dot(rz, ry))
        return yzx
    if mode == 'YXZ':
        yzx = np.dot(rz, np.dot(rx, ry))
        return yzx
    if mode == 'ZXY':
        zxy = np.dot(ry, np.dot(rx, rz))
        return zxy
    if mode == 'ZYX':
        zyx = np.dot(rx, np.dot(ry, rz))
        return zyx
    
def rotation2euler(rotation_matrix, mode):
    r = rotation_matrix
    if mode == 'XYZ':
        alpha = np.arctan2(r[1, 0], r[0, 0])
        beta = np.arcsin(-r[2, 0])
        gamma = np.arctan2(r[2, 1], r[2, 2])
        euler_angles = [alpha, beta, gamma]
        return euler_angles
    if mode == 'XZY':
        alpha = np.arctan2(-r[2, 0], r[0, 0])
        beta = np.arcsin(r[1, 0])
        gamma = np.arctan2(-r[1, 2], r[1, 1])
        euler_angles = [alpha, beta, gamma]
        return euler_angles
    if mode == 'YZX':
        alpha = np.arctan2(r[2, 1], r[1, 1])
        beta = np.arcsin(-r[0, 1])
        gamma = np.arctan2(r[0, 2], r[0, 0])
        euler_angles = [alpha, beta, gamma]
        return euler_angles
    if mode == 'YXZ':
        alpha = np.arctan2(-r[0, 1], r[1, 1])
        beta = np.arcsin(r[2, 1])
        gamma = np.arctan2(-r[2, 0], r[2, 2])
        euler_angles = [alpha, beta, gamma]
        return euler_angles
    if mode == 'ZXY':
        alpha = np.arctan2(r[0, 2], r[2, 2])
        beta = np.arcsin(-r[1, 2])
        gamma = np.arctan2(r[1, 0], r[1, 1])
        euler_angles = [alpha, beta, gamma]
        return euler_angles
    if mode == 'ZYX':
        alpha = np.arctan2(-r[1, 2], r[2, 2])
        beta = np.arcsin(r[0, 2])
        gamma = np.arctan2(-r[0, 1], r[0, 0])
        euler_angles = [alpha, beta, gamma]
        return euler_angles
