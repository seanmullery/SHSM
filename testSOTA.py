"""

This program will test colourisations, produced by various systems against a Ground Truth.
The colourisations must be performed outside of this programme and placed in a folder.
The names of the colourised images must match the names of Ground Truth images.

Seán Mullery

"""

from pathlib import Path
import pandas as pd

import matplotlib.pyplot as plt
from SHSM import*


def main():
    colourisations = []
    file_types = []

    # What colourisation systems do we have? - sort them in alphabetical order
    for folder in sorted(Path('./testImages/colourisationSystems/systems/').glob('*/')):

        if str(folder)[40] != '.':  # deals with /.dstore on Mac
            colourisations.append(str(folder))

    # Get a list of the Ground Truth Image names.
    images = []
    for file in Path('./testImages/colourisationSystems/groundTruth/').glob('*/'):
        if str(file)[44] != '.':  # deals with /.dstore on Mac
            images.append(str(file)[44:-3])  # includes '.' but not extension

    # Make sure to use the right file extension for each colourisation system as some use png and others jpg.
    for i in range(len(colourisations)):
        for types in Path(colourisations[i]).glob(images[0] + '*'):
            file_types.append(str(types)[-3:])

    df = pd.DataFrame(columns=['System', 'Filename', 'L', 'h', 'c', 'Combined'])

    hue_data = np.zeros((20, 7))
    chroma_data = np.zeros((20, 7))
    combined_data = np.zeros((20, 7))
    i = 0
    j = 0
    for system in range(len(colourisations)):

        for image in images:
            gt_bgr = cv2.imread('./testImages/colourisationSystems/groundTruth/' + image + 'jpg')
            cl_bgr = cv2.imread(colourisations[system] + '/' + image + file_types[system])
            cl_bgr = cv2.resize(cl_bgr,(256,256))
            comparison = compare(gt_bgr, cl_bgr)
            temp_list = [colourisations[system][25:], image[:-1],comparison[0], comparison[1],comparison[2],comparison[3]]
            hue_data[i][j] = comparison[1]
            chroma_data[i][j] = comparison[2]
            combined_data[i][j] = comparison[3]
            i = i+1
            df = df.append(pd.Series(temp_list, index=df.columns), ignore_index=True)

        print(colourisations[system][40:])
        i = 0
        j = j+1

    plt.rcParams.update({'font.size': 15})

    # Create a figure instance
    fig1, ax1 = plt.subplots(figsize=(12, 6))

    ax1.set_title('Hue comparison Plot')
    ax1.boxplot(hue_data)
    ax1.set_xticklabels(['ChromaGAN','DeOldify','Iizuka', 'Larsson', 'Nazeri',   'Zhang 1', 'Zhang 2' ] )
    ax1.set_ylabel('SHSM')
    fig1.savefig('./testImages/colourisationSystems/SOTAhue.png', bbox_inches='tight')

    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.set_title('Chroma Comparison Plot')
    ax2.boxplot(chroma_data)
    ax2.set_xticklabels(['ChromaGAN','DeOldify','Iizuka', 'Larsson', 'Nazeri',   'Zhang 1', 'Zhang 2' ])
    ax2.set_ylabel('SSIM')
    fig2.savefig('./testImages/colourisationSystems/SOTAchroma.png', bbox_inches='tight')

    fig3, ax3 = plt.subplots(figsize=(12, 6))
    ax3.set_title('Combined Plot')
    ax3.boxplot(combined_data)
    ax3.set_xticklabels(['ChromaGAN','DeOldify','Iizuka', 'Larsson', 'Nazeri',   'Zhang 1', 'Zhang 2' ])
    ax3.set_ylabel('Combined')
    fig3.savefig('./testImages/colourisationSystems/SOTAcombined.png', bbox_inches='tight')

    df.to_csv('./testImages/colourisationSystems/results.csv')
    print("****************")


if __name__ == "__main__":
    main()
