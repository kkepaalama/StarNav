#/usr/bin/python

### This file contains all the required functions for the global position estimate algorithm

import numpy as np
import math as m

from astropy.time import Time
from astropy.coordinates import SkyCoord
#from astropy.coordinates import ITRS
#from astropy.coordinates import ICRS
import astropy.units as u


#raw image centroid (spherical coordinates)
def raw_cent(time, radec):
    time = [time]
    t = Time(time, format='iso', scale='utc')
    crs = SkyCoord(ra = radec[0]*u.degree, dec = radec[1]*u.degree, obstime = t, frame = 'icrs', unit = 'deg')
    trs = crs.itrs
    trs.representation_type = 'spherical'
    return trs

#raw image centroid (cartesian coordinates)
def cent(time, radec):
    time = [time]
    t = Time(time, format='iso', scale='utc')
    crs = SkyCoord(ra = radec[0]*u.degree, dec = radec[1]*u.degree, obstime = t, frame = 'icrs', unit = 'deg')
    trs = crs.itrs
    trs = np.array([trs.x, trs.y, trs.z])
    return trs


#star vectors (celestial) to star vectors (ecef)
def cel2ecef(time, cel):
    time = [time]
    t = Time(time, format='iso', scale='utc')
    crs = SkyCoord(x = cel[0], y = cel[1], z = cel[2], obstime = t, frame = 'icrs', representation_type = 'cartesian')
    trs = crs.itrs
    ecef = np.array([trs.x, trs.y, trs.z])
    return ecef

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
    R = np.dot(x_rot, np.dot(y_rot, z_rot))
    n_t = np.dot(R, n)
    lat = m.degrees(np.pi/2) - m.degrees(np.arccos(n_t[2]))
    lon = m.degrees(np.arctan2(n_t[1], n_t[0]))
    #if (lon < 0):
    #    lon = lon + 360
    return lat, lon

def R(tilt):
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
    return R

def R_inverse(tilt):
    x_rot = np.array([[1, 0, 0],
                      [0, (np.cos(-tilt[0])), -np.sin(-tilt[0])],
                      [0, np.sin(-tilt[0]), np.cos(-tilt[0])]])
    y_rot = np.array([[np.cos(-tilt[1]), 0, np.sin(-tilt[1])],
                      [0, 1, 0],
                      [-np.sin(-tilt[1]), 0, np.cos(-tilt[1])]])
    z_rot = np.array([[np.cos(-tilt[2]), -np.sin(-tilt[2]), 0],
                      [np.sin(-tilt[2]), np.cos(-tilt[2]), 0],
                      [0, 0, 1]])
    R = np.dot(x_rot, np.dot(y_rot, z_rot))
    return R

def Rot(a, b):
    v = np.cross(a, b)
    s = np.linalg.norm(v)
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