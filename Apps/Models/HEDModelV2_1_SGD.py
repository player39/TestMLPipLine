
import tensorflow as tf
import matplotlib.pyplot as plt
import datetime
import numpy as np
import cv2
from Apps.Model.ModelBase import jyModelBase
from Apps.ModelComponent.HEDComponent import sideBranch, classBalancedSigmoidCrossEntropy
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras import layers, Model, optimizers
from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import state_ops, math_ops


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
        Fuse = layers.Conv2D(1, (1, 1), name='Fuse', padding='SAME', use_bias=False, activation=None, weights=0.2)(Fuse)

        output1 = layers.Activation('sigmoid', name='output1')(Side1)
        output2 = layers.Activation('sigmoid', name='output2')(Side2)
        output3 = layers.Activation('sigmoid', name='output3')(Side3)
        output4 = layers.Activation('sigmoid', name='output4')(Side4)
        output5 = layers.Activation('sigmoid', name='output5')(Side5)
        output6 = layers.Activation('sigmoid', name='output6')(Fuse)

        outputs = [output1, output2, output3, output4, output5, output6]
        self._pModel = Model(inputs=Inputs, outputs=outputs)
        pOptimizer = optimizers.adam(lr=0.0001)
        pOptimizer = optimizers.SGD(lr=0.000001, decay=0., momentum=0.9)
        pOptimizer = monitorSGD(lr=0.000001, decay=0., momentum=0.9)
        # grads = tf.gradients(classBalancedSigmoidCrossEntropy, self._pModel.trainable_weights)
        # pSGD = optimizers.SGD()
        self._pModel.compile(loss={
                                    'output1': classBalancedSigmoidCrossEntropy,
                                    'output2': classBalancedSigmoidCrossEntropy,
                                    'output3': classBalancedSigmoidCrossEntropy,
                                    'output4': classBalancedSigmoidCrossEntropy,
                                    'output5': classBalancedSigmoidCrossEntropy,
                                    'output6': classBalancedSigmoidCrossEntropy
                                   }, optimizer=pOptimizer)
        # self._pModel.summary()

    def startTrain(self, listDS, iMaxLen, iBatchSize):
        itrTrain = tf.compat.v1.data.make_one_shot_iterator(listDS[0])
        itrValid = tf.compat.v1.data.make_one_shot_iterator(listDS[1])

        iStepsPerEpochTrain = int(iMaxLen[0] / iBatchSize[0])
        iStepsPerEpochValid = int(iMaxLen[1] / iBatchSize[1])

        pBack = myCallback(self._strLogPath)

        self._pModel.fit(itrTrain, validation_data=itrValid, epochs=self._iEpochs,
                         callbacks=[self._pSaveModel, self._pTensorboard, pBack], steps_per_epoch=iStepsPerEpochTrain,
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


class myCallback(Callback):
    def __init__(self, path):
        super(myCallback, self).__init__()
        self.pFileWrite = tf.summary.create_file_writer(path + '/metrics')

    def on_epoch_end(self, epoch, logs=None):
        pDict = self.model.optimizer.dictParam[1]
        # t = tf.Variable('training/monitorSGD/gradients/Con1/Conv2D_grad/Conv2DBackpropFilter:0')
        m = K.zeros(pDict.shape)
        p1t = state_ops.assign(m, pDict)
        pUpdate = self.model.optimizer.updates
        for name in pDict:
            with self.pFileWrite.as_default():
                t = pDict[name]['Grad']
                # print(t)
                # iAveGrad = tf.math.reduce_mean(t)
                # tf.summary.scalar('learning rate', iAveGrad, epoch)
                tf.summary.histogram(name, [t], step=epoch)
                # tf.summary.scalar(name, iAveGrad, step=epoch)
            # break


class monitorSGD(optimizers.SGD):
    def __init__(self, lr=0.01, momentum=0., decay=0., nesterov=False, **kwargs):
        super(monitorSGD, self).__init__(lr, momentum, decay, nesterov, **kwargs)
        self.dictParam = []

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [state_ops.assign_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (  # pylint: disable=g-no-augmented-assignment
            1. / (1. + self.decay * math_ops.cast(self.iterations,
                                                  K.dtype(self.decay))))
        # momentum
        shapes = [K.int_shape(p) for p in params]
        moments = [K.zeros(shape) for shape in shapes]
        mg2 = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + moments
        for p, g, m, mg in zip(params, grads, moments, mg2):
            # monitor layer
            self.dictParam.append(self.iterations)
            v = self.momentum * m - lr * g  # velocity
            # mg = K.zeros(g.shape)
            # test = g
            # p = state_ops.ops.Tensor(g, g.shape, dtype=tf.float32)
            self.dictParam.append(math_ops.cast(g, tf.float32))
            #mg2 = K.zeros(g.shape)
            #test2 = g# state_ops.assign(mg2, g)
            self.updates.append(state_ops.assign(m, v))
            # self.dictParam[p.name]['before'] = []
            # self.dictParam[p.name]['after'] = []
            # self.dictParam[p.name]['before'].append(state_ops.assign(mg, test))
            # self.dictParam[p.name]['after'].append(state_ops.assign(mg2, test2))



            # monitor gradient

            # self.dictParam[p.name]['Grad'] = state_ops.assign(mg, g)

            if self.nesterov:
                new_p = p + self.momentum * v - lr * g
                # monitor new_p to p difference
                # self.dictParam[p.name]['Diff'] = state_ops.assign(mg, self.momentum * v - lr * g)
            else:
                # monitor new_p to p difference
                # self.dictParam[p.name]['Diff'] = v
                new_p = p + v

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(state_ops.assign(p, new_p))
        return self.updates


def generateClass():
    return jyHEDModelV1()
