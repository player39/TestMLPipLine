
import tensorflow as tf


# MaxPooling, Input 4-D with shape `[batch, height, width, channels]`
def maxPoolingWithArgmax(featureGroup, stride, padding, poolSize):
    # If Parameter 'include_batch_in_index' is False
    # Return Indices , Value = (y * width + x) * channels + c

    # If Parameter 'include_batch_in_index' is True
    # Value = ((b * height + y) * width + x) * channels + c

    # But This Parameter Not Exist

    # Tips: x, y, c begin zero; width, channels from input shape
    newFeatureGroup, Indices = tf.nn.max_pool_with_argmax(featureGroup, ksize=poolSize, strides=[stride, stride],
                                                          padding=padding)
    Indices = tf.stop_gradient(Indices)
    newFeatureGroup = tf.nn.max_pool(featureGroup, ksize=poolSize, strides=[stride, stride],
                                     padding=padding)

    # print(Indices)
    return newFeatureGroup, Indices


# stride same with input of pooling layer,
def maxUnPooling(featureGroup, Indices, stride, poolSize, padding, outputShape=None):
    inputShape = featureGroup.get_shape().as_list()

    # calculate original size
    Height = (inputShape[1] - 1) * stride + poolSize[0] if outputShape is None else outputShape[1]
    Width = (inputShape[2] - 1) * stride + poolSize[1] if outputShape is None else outputShape[2]
    outputShape = [inputShape[0], Height, Width, inputShape[3]]

    # all elements set zero
    zeroIndices = tf.ones_like(Indices)
    batchRange = tf.reshape(tf.range(outputShape[0], dtype=tf.int64), shape=[outputShape[0], 1, 1, 1])
    b = zeroIndices * batchRange

    # When Parameter 'include_batch_in_index' is False
    y = Indices // (outputShape[2] * outputShape[3])
    x = Indices % (outputShape[2] * outputShape[3]) // outputShape[3]

    # tf.reshape direction is high dimensions to low dimensions
    # test
    # featureRangeTest = tf.range(Indices.get_shape().as_list()[3], dtype=tf.int64)
    featureRange = tf.range(outputShape[3], dtype=tf.int64)
    f = zeroIndices * featureRange

    updateSize = tf.size(featureGroup)
    # test
    # updateSize = tf.size(Indices)
    #updateSize = inputShape[0] * inputShape[1] * inputShape[2] * inputShape[3]
    print(updateSize)
    stack = tf.stack([b, y, x, f])
    # reshape = tf.reshape(tf.stack([b, y, x, f]), [4, updateSize])
    newIndices = tf.transpose(tf.reshape(tf.stack([b, y, x, f]), [4, updateSize]))
    # test
    # featureGroupSize = tf.size(featureGroup)
    value = tf.reshape(featureGroup, [updateSize])
    # value = tf.reshape(featureGroup, [featureGroupSize])

    # return new tensor by newIndices and outputShape
    newFeatureGroup = tf.scatter_nd(newIndices, value, outputShape)
    return newFeatureGroup



'''
test = tf.constant([[
    [[1, 2], [2, 1], [4, 6], [3, 5]],
    [[3, 1], [1, 3], [7, 2], [3, 6]],
    [[2, 1], [4, 3], [1, 2], [5, 2]],
    [[4, 3], [5, 1], [5, 1], [8, 2]],
]])
test2 = tf.constant([[
    [[1, 2], [2, 1], [4, 6], [3, 5]],
    [[3, 1], [1, 3], [7, 2], [3, 6]],
    [[2, 1], [4, 3], [1, 2], [5, 2]],
    [[4, 3], [5, 1], [5, 1], [8, 2]],
]])
t = tf.concat([test, test2], 0)
tt = 0

test = tf.reshape(test, [1, 4, 4, 2])
group, indices = maxPoolingWithArgmax(test, 2, 'SAME', [2, 2])
maxUnPooling(group, indices, 2, [2, 2], 'SAME')

test2 = tf.constant(range(1, 19))
test2 = tf.reshape(test2, [2, 3, 3])
test2 = tf.reshape(test2, [9, 2])
print(test2)
'''
# MaxUnPooling
