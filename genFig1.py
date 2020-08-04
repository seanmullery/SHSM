
from pathlib import Path
import matplotlib.pyplot as plt
from SHSM import*


histo = np.zeros(128)  # This will contain the histogram of the unprocessed images
histo2 = np.zeros((10,128))  # This will contain 10 histograms each processed by a different value of x_0

for x_0 in range(1, 11):
    print(x_0)
    for file in Path('./4000Images/').glob('*.jpg'):
        gt_bgr = cv2.imread('./' + str(file))
        gt_lab = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2LAB)  # Convert BGR to LAB
        gt_comp = np.array(gt_lab[:, :, 1:3], np.float64)   # Convert AB to floats
        gt_comp = (gt_comp[:, :, 0] - 128) + (1j * (gt_comp[:, :, 1] - 128))  # change to complex format centered on 0
        gt_c, gt_h = r2p(gt_comp)  # convert to polar form
        cg1 = 1/(1 + np.exp(-1 * (gt_c - x_0)))  # logistic function

        img1 = ((gt_h / np.pi) + 1) * 127.5
        kernel = np.array([[1, 0, 0],
                           [0, 0, 0],
                           [0, 0, -1]])
        img_der_1 = np.abs(ndimage.filters.convolve(img1, kernel, mode='nearest'))

        img_der_1[img_der_1 > 128] = 256 - img_der_1[img_der_1 > 128]

        kernel = np.array([[0, 0, 1],
                           [0, 0, 0],
                           [-1, 0, 0]])
        img_der_2 = np.abs(ndimage.filters.convolve(img1, kernel, mode='nearest'))

        img_der_2[img_der_2 > 128] = 256 - img_der_2[img_der_2 > 128]
        np.reshape(img_der_1, (256 * 256))

        if x_0 == 1:  # this creates the unprocessed histogram
            hist, bins = np.histogram(img_der_1,bins=128, range=(0,127))
            histo = histo+hist

        img_der_1 = img_der_1 * cg1
        hist, bins = np.histogram(img_der_1, bins=128, range=(0, 127))
        histo2[(x_0 - 1), :] = histo2[(x_0 - 1), :] + hist

        if x_0 == 1:  # this creates the unprocessed histogram
            hist, bins = np.histogram(img_der_2,bins=128, range=(0,127))
            histo = histo+hist

        img_der_2 = img_der_2 * cg1
        hist, bins = np.histogram(img_der_2, bins=128, range=(0, 127))
        histo2[(x_0 - 1), :] = histo2[(x_0 - 1), :] + hist


fig1, ax1 = plt.subplots(figsize=(12,6))

plt.ylim(0,3.0)  # We will only look at 0-4% as this is where most of the information is.
#plt.xlim(0,20)  # Beyond 15 there is little to see.

histo = (histo/8000.0)/65536*100  # average
histo2 = (histo2/8000.0)/65536*100

plt.plot(histo, alpha=1.0, label='Unprocessed' )
plt.plot(histo2[0],alpha=0.4, label='c0=1')
plt.plot(histo2[1],alpha=0.4, label='c0=2')
plt.plot(histo2[2],alpha=0.4, label='c0=3')
plt.plot(histo2[3],alpha=0.4, label='c0=4')
plt.plot(histo2[4],alpha=1.0, label='c0=5')
plt.plot(histo2[5],alpha=0.4, label='c0=6')
plt.plot(histo2[6],alpha=0.4, label='c0=7')
plt.plot(histo2[7],alpha=0.4, label='c0=8')
plt.plot(histo2[8],alpha=0.4, label='c0=9')
plt.plot(histo2[9],alpha=0.4, label='c0=10')


plt.legend(fontsize=18)
plt.xlabel("Gradient Value", fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=18)
plt.ylabel("Average % number of Occurrences", fontsize=18)
plt.title("Histogram of Gradients in Hue Space normalised to 100%", fontsize=24)
fig1.tight_layout()
plt.savefig("fig1.png")
plt.show()








