
import tensorflow as tf


class jyDataSetBase:
    def __init__(self, strFilePath, iMaxLen, iEpoch, iRandomSeed):
        self._strFilePath = strFilePath
        self._iMaxLen = iMaxLen

        self._pTFDataSet = tf.data.TFRecordDataset(strFilePath). \
                                    shuffle(buffer_size=iMaxLen, seed=iRandomSeed). \
                                    repeat(iEpoch). \
                                    map(self.mapFunction)

        self._iTrainLen = None
        self._iValidLen = None
        self._iTestLen = None
        # take function create dataset with at most count elements from this dataset
        self._dictDataSet = {}

        self.strX, self.strY = 'Image', 'Label'
        self.strTrain, self.strValid, self.strTest = 'Train', 'Valid', 'Test'

    def mapFunction(self):
        pass

    def returnDataSet(self):
        pass
