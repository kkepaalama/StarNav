#/usr/bin/python

### Used as a verification script to find flaws in main.py file
### Uses backward propagation and other methods to verify

import numpy as np
import math as m
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.coordinates.representation import CartesianRepresentation
from astropy.coordinates.representation import UnitSphericalRepresentation
from astropy.time import Time
from starimage import StarImage
from coords import s47, s46, s45
from scipy.spatial.transform import Rotation as R


def astro(time, RA, DEC):
    time = [time]
    t = Time(time, format='iso', scale='utc')
    crs = SkyCoord(ra = RA*u.degree, dec = DEC*u.degree, obstime = t, frame = 'icrs', unit = 'deg')
    trs = crs.itrs
    trs.representation_type = 'spherical'
    #trs = np.array([trs.x, trs.y, trs.z])
    return print(trs)

def ost(time, cel):
    time = [time]
    t = Time(time, format='iso', scale='utc')
    crs = SkyCoord(x = cel[0], y = cel[1], z = cel[2], obstime = t, frame = 'icrs', representation_type = 'cartesian')
    trs = crs.itrs
    trs = np.array([trs.x, trs.y, trs.z])
    lat = m.degrees(np.pi/2) - m.degrees(np.arccos(trs[2]))
    lon = m.degrees(np.arctan2(trs[1], trs[0]))
    if (lon < 0):
        lon = lon + 360
    return print(lon,lat)
    
def imu2cam(imu_tilt):
    T = np.array([[-0.0184015, 0.82131394, 0.57017962, -0.07331063],
                  [0.37333988, -0.5233833, 0.76595513, -0.05248031],
                  [0.92751211, 0.22696551, -0.29699821, -0.00126057],
                  [0, 0, 0, 1]])
    cam_tilt = T@imu_tilt
    cam_tilt = np.delete(cam_tilt, 3)
    return cam_tilt

def oste(time, cel, cam_tilt):
    time = [time]
    t = Time(time, format='iso', scale='utc')
    crs = SkyCoord(x = cel[0], y = cel[1], z = cel[2], obstime = t, frame = 'icrs', representation_type = 'cartesian')
    trs = crs.itrs
    trs = np.array([trs.x, trs.y, trs.z])
    x_rot = np.array([[1, 0, 0],
                     [0, np.cos(-cam_tilt[0]), -np.sin(-cam_tilt[0])],
                     [0, np.sin(-cam_tilt[0]), np.cos(-cam_tilt[0])]])
    y_rot = np.array([[np.cos(-cam_tilt[1]), 0, np.sin(-cam_tilt[1])],
                     [0, 1, 0],
                     [-np.sin(-cam_tilt[1]), 0, np.cos(-cam_tilt[1])]])
    z_rot = np.array([[np.cos(cam_tilt[2]), -np.sin(cam_tilt[2]), 0],
                     [np.sin(cam_tilt[2]), np.cos(cam_tilt[2]), 0],
                     [0, 0, 1]])
    n = np.dot(np.dot(x_rot, np.dot(y_rot, z_rot)), trs)
    lat = m.degrees(np.pi/2) - m.degrees(np.arccos(n[2]))
    lon = m.degrees(np.arctan2(n[1], n[0]))
    if (lon < 0):
        lon = lon + 360
    return print(lon,lat)

########
def star(time, cel):
    time = [time]
    t = Time(time, format='iso', scale='utc')
    crs = SkyCoord(x = cel[0], y = cel[1], z = cel[2], obstime = t, frame = 'icrs', representation_type = 'cartesian')
    trs = crs.itrs
    trs = np.array([trs.x, trs.y, trs.z])
    return trs

def cam(time, body):
    time = [time]
    t = Time(time, format='iso', scale='utc')
    crs = SkyCoord(x = body[0], y = body[1], z = body[2], obstime = t, frame = 'icrs', representation_type = 'cartesian')
    body = crs.itrs
    body = np.array([body.x, body.y, body.z])
    return body

def B(body, trs, w):
    body = body.reshape(3,1)
    trs = trs.reshape(3, 1)
    B = np.dot(w, np.dot(body, np.transpose(trs)))
    return B

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
    
def n_s(q):
    n1 = q[0]/(np.sin(np.arccos(q[3])))
    n2 = q[1]/(np.sin(np.arccos(q[3])))
    n3 = q[2]/(np.sin(np.arccos(q[3])))
    return np.array([[n1], [n2], [n3]])

def pos(cam_tilt, n_s):
    x_rot = np.array([[1, 0, 0],
                      [0, np.cos(-cam_tilt[0]), -np.sin(-cam_tilt[0])],
                      [0, np.sin(-cam_tilt[0]), np.cos(-cam_tilt[0])]])
    y_rot = np.array([[np.cos(-cam_tilt[1]), 0, np.sin(-cam_tilt[1])],
                      [0, 1, 0],
                      [-np.sin(-cam_tilt[1]), 0, np.cos(-cam_tilt[1])]])
    z_rot = np.array([[np.cos(cam_tilt[2]), -np.sin(cam_tilt[2]), 0],
                      [np.sin(cam_tilt[2]), np.cos(cam_tilt[2]), 0],
                      [0, 0, 1]])
    n_t = np.dot(np.dot(x_rot, np.dot(y_rot, z_rot)), n_s)
    lat = m.degrees(np.pi/2) - m.degrees(np.arccos(n_t[2]))
    lon = m.degrees(np.arctan2(n_t[1], n_t[0]))
    if (lon < 0):
        lon = lon + 360
    return print(lon, lat)

##########
def A(body, trs, cam_tilt):
    A = trs - body
    x_rot = np.array([[1, 0, 0],
                      [0, np.cos(-cam_tilt[0]), -np.sin(-cam_tilt[0])],
                      [0, np.sin(-cam_tilt[0]), np.cos(-cam_tilt[0])]])
    y_rot = np.array([[np.cos(-cam_tilt[1]), 0, np.sin(-cam_tilt[1])],
                      [0, 1, 0],
                      [-np.sin(-cam_tilt[1]), 0, np.cos(-cam_tilt[1])]])
    z_rot = np.array([[np.cos(cam_tilt[2]), -np.sin(cam_tilt[2]), 0],
                      [np.sin(cam_tilt[2]), np.cos(cam_tilt[2]), 0],
                      [0, 0, 1]])
    A = A.reshape(3, 1)
    A = np.dot(np.dot(x_rot, np.dot(y_rot, z_rot)), A)
    lat = m.degrees(np.pi/2) - m.degrees(np.arccos(A[2]))
    lon = m.degrees(np.arctan2(A[1], A[0]))
    if (lon < 0):
        lon = lon + 360
    return print(lon, lat)



star0 = star(s47.time, s47.cel[0])
star1 = star(s47.time, s47.cel[1])

B0 = B(s47.body[0], star0, s47.wt[0])
B1 = B(s47.body[1], star1, s47.wt[1])
B = B0 + B1

K = K(B)
q = q(K)

r = R.from_quat([q[0], q[1], q[2], q[3]])
r = r.as_matrix()

body0 = np.dot(r, s47.body[0])
body1 = np.dot(r, s47.body[1])

imu_tilt = np.array([0.021997870456728765, 0.0026016829453221896, 0.13, 1])
cam_tilt = imu2cam(imu_tilt)

gp = pos(cam_tilt, body0)


'''##########
#astrometry track
astro = astro(s47.time, 318.582, 19.913)

#openstartracker track 1
ost0 = ost(s47.time, s47.cel[0])

#openstartracker track 2
imu_tilt = np.array([0.021997870456728765, 0.0026016829453221896, 0.13, 1])
cam_tilt = imu2cam(imu_tilt)
oste0 = oste(s47.time, s47.cel[0], cam_tilt)

#full pipeline
star0 = star(s47.time, s47.cel[0])
star1 = star(s47.time, s47.cel[1])
star2 = star(s47.time, s47.cel[2])
star3 = star(s47.time, s47.cel[3])

body0 = cam(s47.time, s47.body[0])
body1 = cam(s47.time, s47.body[1])
body2 = cam(s47.time, s47.body[2])
body3 = cam(s47.time, s47.body[3])

B0 = B(body0, star0, s47.wt[0])
B1 = B(body1, star1, s47.wt[1])
B2 = B(body2, star2, s47.wt[2])
B3 = B(body3, star3, s47.wt[3])
B = B0 + B1 + B2 + B3

K = K(B)
q = q(K)
n_s = n_s(q)
pos = pos(cam_tilt, n_s)

#vector addition
trs = star(s47.time, s47.cel[0])
trs = np.array([trs[0, 0], trs[1, 0], trs[2, 0]])
body = s46.body[0]

A = A(body, trs, cam_tilt)


#astrometry track
astro1 = astro(s46.time, 320.025, 20.126)

#openstartracker track 1
ost01 = ost(s46.time, s46.cel[0])

#openstartracker track 2
imu_tilt1 = np.array([0.02090453337469157, 0.00047506839296058187, 0.0471238898038, 1])
cam_tilt1 = imu2cam(imu_tilt1)
oste01 = oste(s46.time, s46.cel[0], cam_tilt1)

#full pipeline
star01 = star(s46.time, s46.cel[0])
star11 = star(s46.time, s46.cel[1])
star21 = star(s46.time, s46.cel[2])
star31 = star(s46.time, s46.cel[3])

body01 = cam(s46.time, s46.body[0])
body11 = cam(s46.time, s46.body[1])
body21 = cam(s46.time, s46.body[2])
body31 = cam(s46.time, s46.body[3])

B01 = B(body01, star01, s46.wt[0])
B11 = B(body11, star11, s46.wt[1])
B21 = B(body21, star21, s46.wt[2])
B31 = B(body31, star31, s46.wt[3])
B_1 = B01 + B11 + B21 + B31

K1 = K(B_1)
q1 = q(K1)
n_s1 = n_s(q1)
pos1 = pos(cam_tilt1, n_s1)

trs1 = star(s46.time, s46.cel[0])
trs1 = np.array([trs1[0, 0], trs1[1, 0], trs1[2, 0]])
body1 = s46.body[0]

A1 = A(body1, trs1, cam_tilt1)'''










 










