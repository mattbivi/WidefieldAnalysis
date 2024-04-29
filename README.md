# Widefield analysis of Mouse brain imaging
Using the hardware setup as described in (paper), we have developed a processing and visualisation ecosystem in python to allow one to process data exported from the Hamamatsu camera system, after converting to tiff files. 

## Package requirements
Developed in python 3.10 using:
- numpy
- OpenCV (v4.5)
- scipy
- tifffile
- matplotlib
- pandas

for those on unix systems, you can install via 
```
pip3 install numpy OpenCV==4.5.5.64 scipy tifffile matplotlib pandas
```
## Basic pipeline
The stack is designed to be modular, thus we have separated the initial processing, evoked average and visualisation scripts.

### Initial processing
This script is designed to run an entire directory of experiments from one animal over one day *sofie.
This proceeds in the following manner:
- Initially splitting the blue and green channels based on intensity, or loading in separated blue and green tiff files.
- Applying overall motion correction
- Normalising the signal by the average of a neurologically quiet window in the start of the image sequence (often between frames x and y) *sofie
- Cropping the non-brain parts of the image
- Applications of the desired filters (gaussian blur, temporal gaussian blur or low pass filter)
- Global signal regression
- Export as tiff for further processing

### Evoked Average
This script is designed to collate and average the desired experiments placed in a folder to produce a single evoked average of the neurological behaviour.

### ROI fitting
We have written a short script to help fitting the ROIs (Allen Institute Widefield Mouse Brain atlas in folder). It produces a handy GUI that allows the user to quickly fit the brain scan to the desired atlas.
If editing the atlas is required, one can also open them in Fiji or ImageJ.

### ROI valGrabber
Run after the ROI fitting has been done. This generates a csv file containing the average intensity in the ROI fitted in the ROI fitting.
From here, one can either reimport this data into another script or excel to start using the data.

## Further Information
Feel free to forward any questions about the script to mattbivi@gmail.com or annsofie.bjerre@gmail.com for more neuroscience oriented questions.


