import numpy as np
import imutils
import cv2
import os

folder_path = 'Dolo/'
# toFolder = 'Abilify 10 transformed'
toFolder = 'Dolo'
print(os.listdir(folder_path))

allImages = os.listdir(folder_path)


i = 0
for individualImage in allImages:
    if '.DS_Store' not in os.path.basename(individualImage):
        image = cv2.imread(folder_path + os.path.basename(individualImage))
        for angle in np.arange(0, 360, 55):
            rotated = imutils.rotate_bound(image, angle)
            i += 1
            name = str(i) + os.path.basename(individualImage)
            cv2.imwrite(os.path.join(toFolder,name),rotated)

