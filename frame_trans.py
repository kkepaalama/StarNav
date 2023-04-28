#/usr/bin/python

import numpy as np
import math as m

#astropy
from astropy.time import Time
from astropy.coordinates import SkyCoord
import astropy.units as u

#data
from ost_output_papakolea import s1, s2
from main import cel2ecef_car


#ITRS(Cel) to ITRS(ECEF)
def cel2ecef(time, cel):
    time = [time]
    t = Time(time, format='iso', scale='utc')
    crs = SkyCoord(x = cel[0], y = cel[1], z = cel[2], obstime = t, frame = 'icrs', representation_type = 'cartesian')
    trs = crs.itrs
    ecef = np.array([trs.x, trs.y, trs.z])
    return ecef


#ITRS(ECEF)(Geocentric Coordinate Sys) to Local/ENU(East, North, Up)(Topocentric Coordinate Sys)
def ecef2enu(lat, lon, ecef): #where ecef is a 3x1 unit-vector matrix to describe ecef frame
    R1R3 = np.array([[ m.sin(lon),               m.cos(lon),                 0    ]
                     [-m.cos(lon)*m.sin(lat),   -m.sin(lon)*m.sin(lat), m.cos(lat)]
                     [ m.cos(lon)*m.cos(lat),    m.sin(lon)*m.cos(lat), m.sin(lat)]])
    enu = np.dot(R1R3, ecef)
    return enu

#Local/ENU(East, North, Up)(Topocentric Coordinate Sys) to Rover(Body)
def enu2body(tilt): #where tilt is a 3x1 orientation vector gathered from IMU data and OST output
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


rad = np.pi/180
lat = 21*rad #lat = s1.radec[1]
lon = 203*rad #lon = s1.radec[0]

s = cel2ecef_car(s1.time, s1.cel) #vector in ecef
s_ecef = 



