import csv
import time
import math
import IMU
import datetime
import os
import sys
from systemd import daemon
import subprocess


def CaptureImage():
    
    print ("Taking picture")
    timestamp = time.time()
    imagename = srt(time.time())
    cmd = "raspistill -ISO 400 -ss 60000 -awb auto -w 4056 -h 3040 -t 1000 -o /home/pi/Desktop/Kalibr/samples_dynamicIMU/" + imagename + ".png"
    subprocess.call(cmd, shell = True)
    print(cmd)
    print ("Done")

IMU.detectIMU()     #Detect if BerryIMU is connected.
if(IMU.BerryIMUversion == 99):
    print(" No BerryIMU found... exiting ")
    sys.exit()
IMU.initIMU() 

with open('/home/pi/Desktop/Kalibr/imu_dynamic.csv', 'w', encoding = 'UTF8', newline='') as f:
        writer = csv.writer(f)
        header = ['Timestamp', 'ACCx', 'ACCy', 'ACCz', 'GYRx', 'GYRy', 'GYRz']
        writer.writerow(header)
        

while True:#Read the accelerometer,gyroscope and magnetometer values
    ACCx = IMU.readACCx()
    ACCy = IMU.readACCy()
    ACCz = IMU.readACCz()
    GYRx = IMU.readGYRx()
    GYRy = IMU.readGYRy()
    GYRz = IMU.readGYRz()
    timestamp = time.time() #seconds in Unix
 
    header = ['Timestamp', 'ACCx', 'ACCy', 'ACCz', 'GYRx', 'GYRy', 'GYRz']
    data = [[timestamp, ACCx, ACCy, ACCz, GYRx, GYRy, GYRz]]
    
    output = "Timestamp %5.2f " % (timestamp)
    
    if 1:
        output += "#  ACCx %5.2f ACCy %5.2f ACCz %5.2f #  " % (ACCx, ACCy, ACCz)
    if 1:
        output += "\t# GRYx %5.2f  GYRy %5.2f  GYRz %5.2f # " % (GYRx, GYRy, GYRz)
    
    print(output)

    
    if 1:
        output = [timestamp, ACCx, ACCy, ACCz, GYRx, GYRy, GYRz]
    
    CaptureImage()
    
    time.sleep(0.5)
    
    with open('/home/pi/Desktop/Kalibr/imu_dynamic.csv', 'a', encoding = 'UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data)
        
        
    
    







