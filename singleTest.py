"""

This program will test one colourised image against a Ground Truth image

By default it will test the red against the green Mortor and Pestle image
but you can change the images at the command line

Seán Mullery

"""

from SHSM import*
import cv2
import sys
import getopt


def main(argv):
    gt_file = './testImages/MortorPestle/red.png'
    cl_file = './testImages/MortorPestle/green.png'
    try:
        opts, args = getopt.getopt(argv, "hg:c:", ["gt_file=", "cl_file="])
    except getopt.GetoptError:
        print('singleTest.py -g <gt_file> -c <cl_file>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('test.py -g <gt_file> -c <cl_file>')
            sys.exit()
        elif opt in ("-g", "--gt_file"):
            gt_file = arg
        elif opt in ("-c", "--cl_file"):
            cl_file = arg

    gt_bgr = cv2.imread(gt_file)
    gt_bgr = cv2.resize(gt_bgr, (256,256))

    cl_bgr = cv2.imread(cl_file)
    cl_bgr = cv2.resize(cl_bgr, (256,256))

    results = compare(gt_bgr, cl_bgr, create_images=True)
    print('Hue comparison %f' % results[1])
    print('Chroma comparison %f' % results[2])
    print('Combined comparison %f' % results[3])


if __name__ == "__main__":
    main(sys.argv[1:])
