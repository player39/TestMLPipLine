
import cv2
import os


RootPath = 'D:\wyc\Projects\TrainDataSet\HED\Handle\Origin'
generatePath = 'D:\wyc\Projects\TrainDataSet\HED\Handle\enhanceData\Vertical'


def imgFlip():
    listIMG = os.listdir(RootPath)
    for image in listIMG:
        pIMG = cv2.imread(RootPath + '/' + image)
        pGenerateIMG = cv2.flip(pIMG, 0)
        # cv2.imshow('test', pGenerateIMG)
        cv2.imwrite(generatePath + '/vertical_' + image, pGenerateIMG, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

# imgFlip()

def toGray():
    img = cv2.imread('D:\wyc\Projects\TrainDataSet\HED\ComplexData\Train\y/vertical_22.jpg')
    p = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    cv2.imwrite('D:\wyc\Projects\TrainDataSet\HED\ComplexData\Train\y/vertical_22_2.jpg', p, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
toGray()
