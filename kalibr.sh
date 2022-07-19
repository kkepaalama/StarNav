#!/bin/bash

python3 record_imu.py &
echo "Starting image capture in 5 seconds ... press ctrl-c to exit"
for i in `seq 5 -1 1`; do echo $i;sleep 1; done
while true; do
    imagename="/home/pi/Desktop/Kalibr/samples_dynamicIMU/`date +%s.%N`.png"
    raspistill -ISO 400 -ss 60000 -awb auto -w 4056 -h 3040 -t 1000 -o $imagename && echo "Wrote $imagename"
done
