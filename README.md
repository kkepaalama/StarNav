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

## Edit .txt File
The raw output file needs to be edited into a readable format for StarNav.
