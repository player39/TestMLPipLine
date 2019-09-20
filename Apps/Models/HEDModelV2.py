
import tensorflow as tf
import matplotlib.pyplot as plt
import datetime
import numpy as np
import cv2
from Apps.Model.ModelBase import jyModelBase
from Apps.ModelComponent.HEDComponent import sideBranch, classBalancedSigmoidCrossEntropy
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras import layers, Model, optimizers


class jyHEDModelV1(jyModelBase):
    def __init__(self):
        super(jyHEDModelV1, self).__init__()
        self.__listLayerName = []
        self.__pVisualModel = None

    def structureModel(self):
        Inputs = layers.Input(shape=self._inputShape, batch_size=self._iBatchSize)
        Con1 = layers.Conv2D(64, (3, 3), name='Con1', activation='relu', padding='SAME', input_shape=self._inputShape, strides=1)(Inputs)
        Con2 = layers.Conv2D(64, (3, 3), name='Con2', activation='relu', padding='SAME', strides=1)(Con1)
        Side1 = sideBranch(Con2, 1)
        MaxPooling1 = layers.MaxPooling2D((2, 2), name='MaxPooling1', strides=2, padding='SAME')(Con2)
        # outputs1
        Con3 = layers.Conv2D(128, (3, 3), name='Con3', activation='relu', padding='SAME', strides=1)(MaxPooling1)
        Con4 = layers.Conv2D(128, (3, 3), name='Con4', activation='relu', padding='SAME', strides=1)(Con3)
        Side2 = sideBranch(Con4, 2)
        MaxPooling2 = layers.MaxPooling2D((2, 2), name='MaxPooling2', strides=2, padding='SAME')(Con4)
        # outputs2
        Con5 = layers.Conv2D(256, (3, 3), name='Con5', activation='relu', padding='SAME', strides=1)(MaxPooling2)
        Con6 = layers.Conv2D(256, (3, 3), name='Con6', activation='relu', padding='SAME', strides=1)(Con5)
        Con7 = layers.Conv2D(256, (3, 3), name='Con7', activation='relu', padding='SAME', strides=1)(Con6)
        Side3 = sideBranch(Con7, 4)
        MaxPooling3 = layers.MaxPooling2D((2, 2), name='MaxPooling3', strides=2, padding='SAME')(Con7)
        # outputs3
        Con8 = layers.Conv2D(512, (3, 3), name='Con8', activation='relu', padding='SAME', strides=1)(MaxPooling3)
        Con9 = layers.Conv2D(512, (3, 3), name='Con9', activation='relu', padding='SAME', strides=1)(Con8)
        Con10 = layers.Conv2D(512, (3, 3), name='Con10', activation='relu', padding='SAME', strides=1)(Con9)
        Side4 = sideBranch(Con10, 8)
        MaxPooling4 = layers.MaxPooling2D((2, 2), name='MaxPooling4', strides=2, padding='SAME')(Con10)
        # outputs4
        Con11 = layers.Conv2D(512, (3, 3), name='Con11', activation='relu', padding='SAME', strides=1)(MaxPooling4)
        Con12 = layers.Conv2D(512, (3, 3), name='Con12', activation='relu', padding='SAME', strides=1)(Con11)
        Con13 = layers.Conv2D(512, (3, 3), name='Con13', activation='relu', padding='SAME', strides=1)(Con12)
        Side5 = sideBranch(Con13, 16)
        Fuse = layers.Concatenate(axis=-1)([Side1, Side2, Side3, Side4, Side5])

        # learn fusion weight
        Fuse = layers.Conv2D(1, (1, 1), name='Fuse', padding='SAME', use_bias=False, activation=None)(Fuse)

        output1 = layers.Activation('sigmoid', name='output1')(Side1)
        output2 = layers.Activation('sigmoid', name='output2')(Side2)
        output3 = layers.Activation('sigmoid', name='output3')(Side3)
        output4 = layers.Activation('sigmoid', name='output4')(Side4)
        output5 = layers.Activation('sigmoid', name='output5')(Side5)
        output6 = layers.Activation('sigmoid', name='output6')(Fuse)

        outputs = [output1, output2, output3, output4, output5, output6]
        self._pModel = Model(inputs=Inputs, outputs=outputs)
        pAdam = optimizers.adam(lr=0.0001)
        self._pModel.compile(loss={
                                   'output6': classBalancedSigmoidCrossEntropy
                                   }, optimizer=pAdam)

        # self._pModel.summary()

    def startTrain(self, listDS, iMaxLen, iBatchSize):
        itrTrain = tf.compat.v1.data.make_one_shot_iterator(listDS[0])
        itrValid = tf.compat.v1.data.make_one_shot_iterator(listDS[1])

        iStepsPerEpochTrain = int(iMaxLen[0] / iBatchSize[0])
        iStepsPerEpochValid = int(iMaxLen[1] / iBatchSize[1])

        self._pModel.fit(itrTrain, validation_data=itrValid, epochs=self._iEpochs,
                         callbacks=[self._pSaveModel, self._pTensorboard], steps_per_epoch=iStepsPerEpochTrain,
                         validation_steps=iStepsPerEpochValid)

    def loadWeights(self, strPath):
        # last = tf.train.latest_checkpoint(strPath)
        # checkPoint = tf.train.load_checkpoint(strPath)
        self._pModel.load_weights(strPath)
        # visual model
        outputs = []

        for myLayer in self._pModel.layers:
            self.__listLayerName.append(myLayer.name)
            outputs.append(myLayer.output)

        # print(self.__pModel.layers[0])
        # self.__pVisualModel = Model(self.__pModel.inputs, outputs=outputs)
        self.__pVisualModel = Model(self._pModel.inputs, outputs=self._pModel.outputs)
        return self.__pVisualModel

    def predict(self, IMG):
        # pImage = open(IMG, 'rb').read()
        # tensorIMG = tf.image.decode_jpeg(pImage)
        pIMG = image.array_to_img(IMG)# .resize((256, 144))
        tensorIMG = image.img_to_array(pIMG)
        x = np.array(tensorIMG / 255.0)
        # show image
        iColumn = 4
        # generate window
        plt.figure(num='Input')
        # plt.subplot(1, 1, 1)
        plt.imshow(x)

        # imagetest = x

        x = np.expand_dims(x, axis=0)
        # pyplot.imshow(x)
        time1 = datetime.datetime.now()
        outputs = self.__pVisualModel.predict(x)
        time2 = datetime.datetime.now()
        print(time2 - time1)
        i = 100
        listOutput = []
        for i in range(len(outputs)):
            outputShape = outputs[i].shape
            singleOut = outputs[i].reshape(outputShape[1], outputShape[2], outputShape[3])
        # singleOut *= 255
            listOutput.append(singleOut)
        singleOut = listOutput[-1]
        singleOut[singleOut > 0.5] = 1
        listOutput[-1] = singleOut
        return listOutput
        '''
        for output in outputs:
            # plt.figure(num='%s' % str(i))
            outputShape = output.shape
            singleOut = output.reshape(outputShape[1], outputShape[2], outputShape[3])
            singleOut *= 255
            if outputShape[3] == 1:
                # test = x - output
                # test = np.abs(test)
                # return mysum

                # plt.subplot(1, 1, 1)
                # plt.imshow(singleOut, camp='gray')
                # cv2.imwrite('D:\wyc\Projects\TrainDataSet\HED\Result/%s.jpg' % str(i), singleOut)
                return singleOut
                # i += 1
                # plt.show()
        '''
    def getModelConfig(self):
        return self._iBatchSize


def generateClass():
    return jyHEDModelV1()
