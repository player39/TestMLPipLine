
import os
import cv2

path = 'D:\wyc\Projects\TrainDataSet\HED\ComplexDataBlack\Validation\y'
for image in os.listdir(path):
    pIMG = cv2.imread(path + '/' + image)
    pIMG = cv2.cvtColor(pIMG, cv2.COLOR_RGB2GRAY)
    width = pIMG.shape[0]
    height = pIMG.shape[1]
    for i in range(width):
        for j in range(height):
            pIMG[j][i] = 255.0 - pIMG[j][i]
    cv2.imwrite(path + '/' + image, pIMG)
