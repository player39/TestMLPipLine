
import datetime
import configparser
import os
import tensorflow as tf
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard
# from keras.callbacks import TensorBoard
from Apps.Settings.Settings import ProjectPath, strModelFileName


class jyModelBase:
    def __init__(self):
        self._pModel = None
        self._strModelName = strModelFileName
        self._pConfig = configparser.ConfigParser()
        strConfigPath = os.path.dirname(__file__) + '/Config/%s' % self._strModelName
        if not os.path.exists(strConfigPath):
            os.mkdir(strConfigPath)
        # strConfigPath = '/Config/%s.ini' % self._strModelName
        self._pConfig.read(strConfigPath + '/%s.ini' % self._strModelName)

        strEpochs = 'Epochs'
        strBatch = 'Batch'
        strInputShape = 'InputShape'
        strSavePeriod = 'SavePeriod'
        if not self._pConfig.has_section(self._strModelName):
            self._pConfig[self._strModelName] = {
                'Epochs': 100,
                'Batch': 32
            }

        if not self._pConfig.has_option(self._strModelName, strEpochs):
            self._pConfig[self._strModelName][strEpochs] = 100
        self._iEpochs = int(self._pConfig[self._strModelName][strEpochs])

        if not self._pConfig.has_option(self._strModelName, strBatch):
            self._pConfig[self._strModelName][strBatch] = 32
        self._iBatchSize = int(self._pConfig[self._strModelName][strBatch])

        if not self._pConfig.has_option(self._strModelName, strInputShape):
            self._pConfig[self._strModelName][strInputShape] = '224,224,3'
        self._inputShape = [int(shape) for shape in self._pConfig[self._strModelName][strInputShape].split(',')]
        self._inputShape = tuple(self._inputShape)

        if not self._pConfig.has_option(self._strModelName, strSavePeriod):
            self._pConfig[self._strModelName][strSavePeriod] = '5'
        self._iPeriod = int(self._pConfig[self._strModelName][strSavePeriod])

        with open(strConfigPath + '/%s.ini' % self._strModelName, 'w') as configfile:
            self._pConfig.write(configfile)

        self._strFormat = '--------------%s--------------'
        self._strModelFileName = '/cp-{epoch:04d}.ckpt'

        self._strSavePath = ProjectPath + '\Model\Save\%s\CheckPoint\%s' % (self._strModelName, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        self._pSaveModel = ModelCheckpoint(self._strSavePath + '/cp-{epoch:04d}.ckpt',
                                           save_weights_only=True, verbose=1, period=self._iPeriod)
        self._strLogPath = ProjectPath + '\Model\Save\%s\Log\%s' % (self._strModelName, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        self._pTensorboard = TensorBoard(log_dir=self._strLogPath, histogram_freq=1, write_images=True, write_grads=True)

    def structureModel(self):
        pass

    def startTrain(self, listDS, iMaxLen, iBatchSize):
        pass

    def predict(self, IMG):
        pass

    def loadWeights(self, strPath):
        pass
