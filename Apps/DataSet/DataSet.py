
import tensorflow as tf


class jyDataSet:
    def __init__(self, strFilePath, iMaxLen, iEpoch, iRandomSeed=10):
        self.__strFilePath = strFilePath
        self.__iMaxLen = iMaxLen

        self.__pTFDataSet = tf.data.TFRecordDataset(strFilePath). \
                                    shuffle(buffer_size=iMaxLen, seed=iRandomSeed). \
                                    repeat(iEpoch). \
                                    map(self.mapFunction)

        self.__iTrainLen = None
        self.__iValidLen = None
        self.__iTestLen = None
        # take function create dataset with at most count elements from this dataset
        self.__dictDataSet = {}

        self.strX, self.strY = 'Image', 'Label'
        self.strTrain, self.strValid, self.strTest = 'Train', 'Valid', 'Test'

    def mapFunction(self, exampleProto):
        dictFeatures = {
            'x': tf.io.FixedLenFeature([], tf.string),
            'y': tf.io.FixedLenFeature([], tf.string)
        }
        parsedFeatures = tf.io.parse_single_example(exampleProto, dictFeatures)
        tensorXIMG = tf.image.decode_jpeg(parsedFeatures['x'])
        # test = tf.
        # casts tensor to a new type
        tensorXIMG = tf.cast(tensorXIMG, tf.float32)
        tensorXIMG /= 255.0
        # tensorIMG = np.expand_dims(tensorIMG, axis=0)
        parsedFeatures['x'] = tensorXIMG

        tensorYIMG = tf.image.decode_jpeg(parsedFeatures['y'])
        tensorYIMG = tf.cast(tensorYIMG, tf.float32)
        tensorYIMG /= 255.0

        parsedFeatures['y'] = tensorYIMG
        return parsedFeatures['x'], parsedFeatures['y'] # , parsedFeatures['y'], parsedFeatures['y'], parsedFeatures['y'], parsedFeatures['y'], parsedFeatures['y'])

    def returnDataSet(self):
        return [self.__pTFDataSet, self.__iMaxLen]


