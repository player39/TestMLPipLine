
import tensorflow as tf
import numpy as np
from keras import backend as k
from tensorflow.python.layers import layers
from tensorflow.python.keras import initializers
from tensorflow.python.keras.regularizers import l1, l2


def upsampling_bilinear(fac, num, chan):
    # 确定卷积核大小
    def get_kernel_size(factor):
        return 2 * factor # -factor % 2
    # 创建相关矩阵

    def upsample_filt(size):
        factor = (size+1)//2
        if size % 2 == 1:
            center = factor-1
        else:
            center = factor-0.5
        og = np.ogrid[: size, : size]

        return (1 - abs(og[0]-center) / factor) * (1 - abs(og[1] - center) / factor)
    # 进行上采样卷积核

    def bilinear_upsample_weights(factor):
        filter_size = get_kernel_size(factor)
        weights = np.zeros((filter_size, filter_size), dtype=np.float32)
        upsample_kernel = upsample_filt(filter_size)
        # print(upsample_kernel)
        weights[:, :] = upsample_kernel
        return weights
    weights = bilinear_upsample_weights(fac)
    return weights

#t = upsampling_bilinear(4, 1, 1)
#r = 1


def sideBranch(input, factor):
    Con1 = layers.Conv2D(1, (1, 1), activation=None, padding='SAME', kernel_regularizer=l2(0.00001))(input)
    kernelSize = (2 * factor, 2 * factor)
    initWeight = upsampling_bilinear(factor, 1, 1)
    # initWeight = tf.initializers.Constant(value=initWeight)
    initWeight = initializers.constant(value=initWeight)
    # initializer = tf.constant_initializer(value=initWeight)
    # initWeight = tf.Variable(initial_value=initializer(shape=kernelSize, dtype=tf.float32))
    # p = layers.Conv2DTranspose(1, kernelSize, strides=factor, padding='SAME', use_bias=False, activation=None, weights=initWeight)
    # p.set_weights(initWeight)
    # t = p.weights

    DeCon = layers.Conv2DTranspose(1, kernelSize, strides=factor, padding='SAME', use_bias=False, activation=None, kernel_initializer=initWeight, kernel_regularizer=l2(0.00001))(Con1)
    # test = p.weights
    return DeCon

'''
def classBalancedSigmoidCrossEntropy(logits, label):
    y = tf.cast(label, tf.float32)
    # percentage of positive data
    countPos = tf.reduce_sum(y)
    # percentage of negative
    countNeg = tf.reduce_sum(1.0 - y)
    beta = countNeg / (countNeg + countPos)
    tf.print(logits)
    posWeight = tf.math.log(beta / (1 - beta))
    cost = tf.nn.weighted_cross_entropy_with_logits(logits, y, posWeight)
    cost = tf.reduce_mean(cost * (1 - beta))

    return cost
'''


def classBalancedSigmoidCrossEntropy(y_true, y_pred):
    """
    Implements Equation [2] in https://arxiv.org/pdf/1504.06375.pdf
    Compute edge pixels for each training sample and set as pos_weights to tf.nn.weighted_cross_entropy_with_logits
    """
    # Note: tf.nn.sigmoid_cross_entropy_with_logits expects y_pred is logits, Keras expects probabilities.
    # transform y_pred back to logits

    # due to log(y_pred) y_pred not equal 1 or 0
    _epsilon = _to_tensor(k.epsilon(), y_pred.dtype.base_dtype)
    y_pred = tf.clip_by_value(y_pred, _epsilon, 1 - _epsilon)

    # tf.nn.sigmoid_cross_entropy_with_logits 需求的logits是不经过sigmoid函数, SideOut层的输出, 但是定义网络结构的时候
    # y_pred已经通过sigmoid函数 因此接下来的这句指令是执行了sigmoid逆过程求出SideOut层输出, 实际上这里非常多余
    # 定义网络结构时可以删去5个Activation
    y_pred = tf.math.log(y_pred / (1 - y_pred))

    y_true = tf.cast(y_true, tf.float32)

    count_neg = tf.reduce_sum(1. - y_true)
    count_pos = tf.reduce_sum(y_true)

    # Equation [2]
    beta = count_neg / (count_neg + count_pos)

    # Equation [2] divide by 1 - beta
    pos_weight = beta / (1 - beta)

    cost = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, targets=y_true, pos_weight=pos_weight)

    # Multiply by 1 - beta
    cost = tf.reduce_mean(cost * (1 - beta))

    # check if image has no edge pixels return 0 else return complete error function
    return tf.where(tf.equal(count_pos, 0.0), 0.0, cost)


def ofuse_pixel_error(y_true, y_pred):
    pred = tf.cast(tf.greater(y_pred, 0.5), tf.int32, name='predictions')
    error = tf.cast(tf.not_equal(pred, tf.cast(y_true, tf.int32)), tf.float32)
    return tf.reduce_mean(error, name='pixel_error')


def _to_tensor(x, dtype):
    """Convert the input `x` to a tensor of type `dtype`.
    # Arguments
    x: An object to be converted (numpy array, list, tensors).
    dtype: The destination type.
    # Returns
    A tensor.
    """
    x = tf.convert_to_tensor(x)
    if x.dtype != dtype:
        x = tf.cast(x, dtype)
    return x
