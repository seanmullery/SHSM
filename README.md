# SHSM

The SHSM directory is where the important functions are stored.  All other
files in this repository are for testing/demonstrating the functionality.
Our paper shows and discusses the results of these.


To do a single test of a ground truth image against a colourised image, use
singleTest.py

usage: python singleTest.py -g <ground_truth_image> -c <colourised_image>

This will print out the comparison result to the console and will also create
a map image showing where the images differ, as you see from the example below.

<img src='https:/github.com/seanmullery/SHSM/SSIM_SHSM_map.png' width=800>
