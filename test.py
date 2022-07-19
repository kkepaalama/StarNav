import time
import sys
from systemd import daemon
import subprocess


def CaptureImage():
    
    print ("Taking picture")
    timestamp = time.time()
    imagename = time.strftime('%s')
    cmd = "raspistill -ISO 400 -ss 60000 -awb auto -w 4056 -h 3040 -t 1000 -o /home/pi/Desktop/Kalibr/test_samples/" + imagename + ".png"
    subprocess.call(cmd, shell = True)
    print(cmd)
    print ("Done")

CaptureImage()

