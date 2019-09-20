
import importlib
import os
import configparser
from Apps.Settings.Settings import strModelFileName, strDataSetRootPath
from Apps.DataSet.DataSet import jyDataSet


class jyPipeLine:
    def __init__(self):
        # 根目录
        self.__strRootPath = os.path.abspath(os.path.dirname(os.getcwd()))
        # Model目录下文件名
        # 判断Model是否存在, 不存在则创建
        if not os.path.exists(self.__strRootPath + '/Model'):
            os.mkdir('/Model')
        listModelFileName = os.listdir(self.__strRootPath + '/Model')
        # 动态加载Model更改配置文件则加载不同的Model
        if strModelFileName + '.py' not in listModelFileName:
            print('Model File Not Exists')
        moduleModel = importlib.import_module('Apps.Model.%s' % strModelFileName)
        # Model
        self.__pModel = moduleModel.generateClass()

    def train(self):
        # DSName
        strDSName = 'ComplexDS'
        # DataSet
        strDSPath = strDataSetRootPath + '/TFRecordBlack/'
        listFolder = []
        # listLen = [1344, 1080, 1682, 1588, 1678, 1721, 1688]
        # iSum = sum(listLen)
        # listLen.clear()
        # listLen.append(iSum)
        '''
        listPath = []
        if len(listFolder):
            for folder in listFolder:
                listPath.append(strDSPath + '/%s/TFRecord/Normal.tfrecord' % folder)
        else:
            listPath.append(strDSPath + '/TFRecord/Normal.tfrecord')
        '''
        listFolder = ['ComplexData']
        # strDSName = 'HED'
        strTrainPath = strDataSetRootPath + '/%s'
        iTrain = 0
        iValid = 0
        for folder in listFolder:
            iTrain += len(os.listdir(strTrainPath % folder + '/Train/x'))
            iValid += len(os.listdir(strTrainPath % folder + '/Validation/x'))

        # iTrain = 1344 + 1352 + 1344 + 1264 + 1336 + 1008 + 1328
        # iValid = 345 + 340 + 338 + 325 + 272 + 272 + 344
        self.__pDS = jyDataSet(strDSPath + 'Train.tfrecord', iTrain, 20000)
        # self.__pDS.splitTFDataSet([8, 2, 0])
        pValidDS = jyDataSet(strDSPath + 'Valid.tfrecord', iValid, 20000)
        iBatch = 8
        TrainDS = self.__pDS.returnDataSet()
        TrainDS[0] = TrainDS[0].batch(iBatch)
        ValidDS = pValidDS.returnDataSet()
        ValidDS[0] = ValidDS[0].batch(iBatch)
        # self.__pDS.returnDataSet().
        # self.__pModel.structureModel()
        self.__pModel.startTrain([TrainDS[0], ValidDS[0]], [TrainDS[1], ValidDS[1]], [iBatch, iBatch])

    def predict(self, image):
        generateFrame = self.__pModel.predict(image)
        return generateFrame

    def loadWeights(self):
        self.__pModel.structureModel()
        self.__pModel.loadWeights('D:\wyc\Projects\TestPipeLine\Apps\Model\Save\HEDModelV2_2_SGD_GradientTape_L1\CheckPoint/20190816-180133/cp-16921.ckpt')
        self.__pModel.generateVisualModel()
        print(1)


test = jyPipeLine()

test.loadWeights()
# test.generateVisualModel()

from LibApps.HikVideoStream.ReadStreamCV2 import jyReadStreamCV2Base
import cv2
video = jyReadStreamCV2Base('Belt1')
video.startCaptureVideo()
listFrame = video.getListFrame()

while True:
    i = 0
    iLen = len(listFrame)
    while i < iLen:
        frames = test.predict(listFrame[i])
        for J in range(len(frames)):
            cv2.imshow('line%s' % str(J), frames[J])
        # cv2.imshow('line', frame)
        listFrame.pop(i)
        iLen -= 1
        if cv2.waitKey(1) & 0xff == ord('q'):
            break

test.train()

