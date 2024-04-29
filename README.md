# StarNav

## General Overview
StarNav is an open-source algorithm which offers solutions to global positioning using celestial techniques and sensor measurements. This code package is an extension of OpenStartracker, generating lost in space identification, and performs most of the heavy lifting regarding image processing.

## Harware Setup
StarNav was intended to be integrated with a sensor suite capable of taking stellar images as well as measuring a gravity vector. The low-cost sensor suite is comprised of four main components; Raspberry Pi High Quality Camera, 16mm 12MP Telephoto Lens, Raspberry Pi 4, and a BerryIMU. These components make up a star tracker sensor suite specifically for global positioning.

## Field Testing
It is important to capture images with a clear view of the stars. Therefore, field tests should only be conducted on clear nights in areas with low lighting. To reduce blur, ensure that the sensor suite remains static.

## Basic Setup
- Install OpenStartracker using linux software. For Windows users, install Windows Subsystem for Linux (WSL). For step-by-step instructions, please navigate to https://github.com/UBNanosatLab/openstartracker?tab=readme-ov-file.
- Run OpenStartracker using example imagery. OpenStartracker provides sample images to test calibration process.

## Code Execution 
- Create a new directory to calibrate a new camera set up.
- Upload new stellar images taken with the sensor suite to this new directory.
- Replace the ```startracker.py``` in OpenStarTracker with the modified version listed above.
- Run ```cd tests/ ./unit_test.sh -crei yourcamera```
- **Note** Should the images fail to solve, edit the photos by varying the contrast. The images should clearly show bright starts against a dark background.
- Save raw output as a .txt file.
- Edit .txt file into the correct format.
- Run ```base.py``` using ```StarImage``` class.

## Edit .txt File
The raw output file needs to be edited into a readable format for StarNav. Edit the file to match the format below. 
~~~
time = 'YYYY-MM-DD HH:MM:SS' #rgb.solve_image('directory/image.png')
DEC=21.380656
RA=280.552804
ORIENTATION=-166.918234
stars =  1
body = [[0.9953135251998901, -0.04998224973678589, 0.08278121799230576]]
eci = [[0.2399647980928421, -0.9237858057022095, 0.29839015007019043]]
wt = [1.3923705392478263]
radec = [RA, DEC]
body = np.array(body)
eci = np.array(eci)
tilt = [-0.005935353, -0.014718526, np.radians(180 + ORIENTATION)]
s1 = StarImage(time, body, eci, wt, tilt, stars, radec)
~~~
The tilt measurements will need to matched to each corresponding image and recorded in the text. Due to the exposure time of the camera on the sensor suite, 6 seconds (the duration of exposure) was added to ```time```. It is also worth noting that the camera default setting timestamps image in UTC+0:00. Edit ```time``` based on the timezone the image was taken in. Editing the raw output file into this format will create an object in the ```StarImage``` class.






