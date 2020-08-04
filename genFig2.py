
from pathlib import Path
import matplotlib.pyplot as plt
from SHSM import*


def logistic(x):
    return 1/(1+np.exp(-1*(x-5)))


def plot_mean_and_ci(mean, lb, ub, ax, color_mean=None, color_shading=None):

    ax.fill_between(range(mean.shape[0]), ub, lb, color=color_shading, alpha=0.5)
    ax.plot(mean, color_mean)


histo = np.zeros((4000,128))  # This will contain 4000 histograms, one for each image

x_0 = 5  # The chosen value from figure 1
img_index = 0

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

    img_der_1 = img_der_1 * cg1
    hist, bins = np.histogram(img_der_1, bins=128, range=(0, 127))
    histo[img_index, :] = histo[img_index, :]+hist

    img_der_2 = img_der_2 * cg1
    hist, bins = np.histogram(img_der_2, bins=128, range=(0, 127))
    histo[img_index, :] = histo[img_index, :]+hist
    img_index = img_index + 1

histo = (histo/2.0)/65536*100

histo_means = np.mean(histo, axis=0)
histo_sds = np.std(histo, axis=0)
fig1, ax1 = plt.subplots(figsize=(12, 6))
plot_mean_and_ci(histo_means, histo_means - histo_sds, histo_means + histo_sds, ax1, color_mean='k', color_shading='k')

plt.xlim(0, 15)  # Beyond 15 there are few occurrences
color = 'k'
ax1.tick_params(axis='y', labelcolor=color, which='major', labelsize=18)
ax1.tick_params(axis='x', labelcolor=color, which='major', labelsize=18)

plt.xlabel("Gradient Value", fontsize=18)
plt.ylabel("Percentage of pixels", fontsize=18)
plt.title("Histogram of Gradients in Hue Space, normalised to 100%", fontsize=24)

x = np.linspace(0, 15, num=16)

y = logistic(x)
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:Red'
ax2.set_ylabel('Output Gradient Value', color=color, fontsize=18)  # we already handled the x-label with ax1
ax2.plot(x,y, color=color, linewidth=3)

ax2.tick_params(axis='y', labelcolor=color, which='major', labelsize=18)
fig1.tight_layout()


plt.axvspan(0, 1, color='y', alpha=0.2, lw=0)
plt.axvspan(1, 9, color='g', alpha=0.2, lw=0)
plt.axvspan(9, 15, color='b', alpha=0.2, lw=0)

plt.annotate('Hue Texture Pixels', (3,0.5), fontsize=20, alpha=0.5)
plt.annotate('Hue Edge Pixels', (10.5,0.5), fontsize=20, alpha=0.5)

plt.savefig("fig2.png")
plt.show()

