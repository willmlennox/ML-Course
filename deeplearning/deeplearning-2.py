import numpy as np
import matplot.pyplot as plt
import os
import cv2
from tqdm import tqdm

DATADIR = "C:/Users/sweet/Downloads/kagglecatsanddogs_3367a/PetImages"

CATEGORIES = ["Dog", "Cat"]

IMG_SIZE = 100

for category in CATEGORIES:
	path = os.path.join(DATADIR, category)
	for img in os.listdir(path):
		img_array = cv2.imread(os.path.join(path,img) , cv2.IMREAD_GRAYSCALE)
		plt.imshow(img_array, cmap='gray')
		plt.show()

		break
	break


