#/usr/bin/python

### Workpad used to execute main.py

from coords_papakolea import s1
from main import raw_cent, cent, cel2ecef, B, K, q, n, imu2cam, pos
import numpy as np
from scipy.spatial.transform import Rotation as R


raw_cent = raw_cent(s1.time, s1.radec)
cent = cent(s1.time, s1.radec)

star0 = cel2ecef(s1.time, s1.cel[0])
star1 = cel2ecef(s1.time, s1.cel[1])
star2 = cel2ecef(s1.time, s1.cel[2])
star3 = cel2ecef(s1.time, s1.cel[3])
star4 = cel2ecef(s1.time, s1.cel[4])
star5 = cel2ecef(s1.time, s1.cel[5])
star6 = cel2ecef(s1.time, s1.cel[6])
star7 = cel2ecef(s1.time, s1.cel[7])
star8 = cel2ecef(s1.time, s1.cel[8])
star9 = cel2ecef(s1.time, s1.cel[9])
star10 = cel2ecef(s1.time, s1.cel[10])
star11 = cel2ecef(s1.time, s1.cel[11])

body0 = cel2ecef(s1.time, s1.body[0])
body1 = cel2ecef(s1.time, s1.body[1])
body2 = cel2ecef(s1.time, s1.body[2])
body3 = cel2ecef(s1.time, s1.body[3])
body4 = cel2ecef(s1.time, s1.body[4])
body5 = cel2ecef(s1.time, s1.body[5])
body6 = cel2ecef(s1.time, s1.body[6])
body7 = cel2ecef(s1.time, s1.body[7])
body8 = cel2ecef(s1.time, s1.body[8])
body9 = cel2ecef(s1.time, s1.body[9])
body10 = cel2ecef(s1.time, s1.body[10])
body11 = cel2ecef(s1.time, s1.body[11])


B0 = B(body0, star0, s1.wt[0])
B1 = B(body1, star1, s1.wt[1])
B2 = B(body2, star2, s1.wt[2])
B3 = B(body3, star3, s1.wt[3])
B4 = B(body4, star4, s1.wt[4])
B5 = B(body5, star5, s1.wt[5])
B6 = B(body6, star6, s1.wt[6])
B7 = B(body7, star7, s1.wt[7])
B8 = B(body8, star8, s1.wt[8])
B9 = B(body9, star9, s1.wt[9])
B10 = B(body10, star10, s1.wt[10])
B11 = B(body11, star11, s1.wt[11])
B = B0 + B1 + B2 + B3 + B4 + B5 + B6 + B7 + B8 + B9 + B10 + B11

K = K(B)
q = q(K)
n = n(q)

r = R.from_quat([q[0], q[1], q[2], q[3]])
r = r.as_matrix()

body0 = np.dot(r, body0)
body1 = np.dot(r, body1)
body2 = np.dot(r, body2)
body3 = np.dot(r, body3)
body4 = np.dot(r, body4)
body5 = np.dot(r, body5)
body6 = np.dot(r, body6)
body7 = np.dot(r, body7)
body8 = np.dot(r, body8)
body9 = np.dot(r, body9)
body10 = np.dot(r, body10)
body11 = np.dot(r, body11)
center = np.dot(r, cent)

imu_tilt = s1.imu_tilt
#cam_tilt = imu2cam(imu_tilt)
cam_tilt = np.array([0, 0, 0])

pos_0 = pos(cam_tilt, body0)
pos_1 = pos(cam_tilt, body1)
pos_2 = pos(cam_tilt, body2)
pos_3 = pos(cam_tilt, body3)
pos_4 = pos(cam_tilt, body4)
pos_5 = pos(cam_tilt, body5)
pos_6 = pos(cam_tilt, body6)
pos_7 = pos(cam_tilt, body7)
pos_8 = pos(cam_tilt, body8)
pos_9 = pos(cam_tilt, body9)
pos_10 = pos(cam_tilt, body10)
pos_11 = pos(cam_tilt, body11)
centroid = pos(cam_tilt, cent)