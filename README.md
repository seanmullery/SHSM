# SHSM

The SHSM directory is where the important functions are stored.  All other
files in this repository are for testing/demonstrating the functionality.
Our paper shows and discusses the results of these.

# Single Test
To do a single test of a ground-truth image against a colourised image, use
singleTest.py

usage:
```shell
python singleTest.py -g <ground_truth_image> -c <colourised_image>
```


This will print out the comparison result to the console and will also create
a map image showing where the images differ, as you see from the example below.
The map file will be called SSIM_SHSM_map and will be saved in the root directory.


If you don't specify your own images, it will default to using the Mortar and
Pestle images.

<img src='./SSIM_SHSM_map.png' width=800>


# Testing multiple colourisations against a single ground-truth.

To do this you should use the testRecolourisations.py program.

usage:
```shell
python testRecolourisations.py
```

This will use the folder called "recolour". In this folder is placed a
ground-truth image called GTP.png. There is also a folder of colourisations
called "colourisations". This contains all the colourisations that we want to
compare with our ground-truth image.

If you want to test your own you will need to rename your ground-truth image to
GTP.png but the images in the colourisations folder can be anything as long as
they are png files. To use jpg files you will simply have to modify the testRecolourisations.py file to reflect this.

This will output a csv file called psRecolourResults.csv which will contain
a row of results for each image in the colourisations folder.


# Testing SOTA colourisation algorithms.

To do this we used 20 images from the places dataset. Ideally we would choose more
but many of the SOTA systems we wanted to test require manually uploading a grey-scale
prior into a web interface one at a time and then downloading the resulting
colourisation. Despite the small number of images we found that there was still
a large variance in the results.

To try this yourself you will need to use the file testSOTA.py

usage:

```shell
python testSOTA.py
```

This will use the folder "./testImages/colourisationSystems/".
In this folder we have put the 20 ground-truth images in a folder called "groundTruth".
We have put the colourisations of the 20 images in a folder called "colourisations"
and we have a separate sub-folder under "colourisations" for each system that we
tested.
Note that the names of each image must be the same as that image in the "groundTruth"
folder. Because some systems output .jpg and some output .png, we have allowed for both.
Do keep in mind that the quality of compression used will effect the results.

To create your own grey-scale prior you can simply convert the colour ground-truth
images to the CIEL*a*b* space and use only the L-channel.


# SHSM library
The file SHSM.py contains the library functions for SHSM.
It contains the SSIM function for standard SSIM which is used the the chroma comparison.
The SHSM function for the Hue comparison.

A function called r2p or changing from cartesian to polar form. We take the CIEL*a*b* space and with a+jb we  change to c &angle; h.

A function called compare. This function will take the BGR images, convert them to Lhc and send each to the appropriate comparison function (SSIM or SHSM). It will report the comparison values as a list [L-comparison, h-comparison, c-comparison, combined h*c comparison]. The L-comparison is rarely used but can give an indication of whether
a lot of deterioration has occurred to the file during processing. For example,
if a high-compression rate is used then this will show up as a difference between
the L-channel of the ground-truth and the L-channel of the colourisation. We can also expect that some deterioration will occur from quantization while changing between BGR - CIEL*a*b* - Lhc - CIEL*a*b* - BGR.
When calling the compare function you can specify that you want to create output
images, which is set to False by default. This will produce the output map
(used above in the single test example) but will only create one output image called
SSIM_SHSM_map.png. This means that if you call the compare function multiple times, as
we do in the testSOTA.py and testRecolourisations.py it will continuously overwrite
this file but will also slow the algorithm down considerably while doing this.
It would not take much coding effort to modify this to your desired implementation
if you wished to change the file name or output multiple files as maps.



# Hyper-parameter choices
Two files "genFig1.py" and "genFig2.py" are used to generate Figures 1 & 2 from
our paper. These helped us to determine the best value for hyper-parameters.
However for this we used 4000 images from the places dataset. We have included the
directory structure for this, but not the actual files. You can re-create this
experiment with files of your own but there are are hard-coded numbers in the files
which assue 4000 images of 256 x 256 were used. If you change the number or size of
the images you should change these numbers. We would expect that whatever images you
use, as long as you have a significant number of them and they are natural images,
you should see similar results.
