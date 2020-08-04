"""

This program will test images that have been re-colourised in photoshop, against a Ground Truth.
The colourisations must be performed outside of this programme and placed in a folder.

Seán Mullery

"""

from pathlib import Path
import pandas as pd
from SHSM import*


def main():

    # Get a list of the Ground Truth Image names.
    images = []
    for file in sorted(Path('./testImages/recolour/colourisations').glob('*/')):  # sort in alphabetical order
        if str(file)[35] != '.':  # deals with /.dstore on Mac
            images.append(str(file))

    df = pd.DataFrame(columns=['Filename', 'L', 'h', 'c','Combined h*c'])

    for image in images:
        gt_bgr = cv2.imread('./testImages/recolour/GTP.png')
        cl_bgr = cv2.imread(image)
        cl_bgr = cv2.resize(cl_bgr, (256, 256))
        comparison = compare(gt_bgr, cl_bgr)
        temp_list = [image, comparison[0], comparison[1], comparison[2], comparison[3]]
        df = df.append(pd.Series(temp_list, index=df.columns), ignore_index=True)

    df.to_csv('psRecolourResults.csv')

    print("****************")


if __name__ == "__main__":
    main()
