import  tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Activation,
    Dense,
    Flatten,
    Dropout,
    UpSampling2D
)
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
)
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2

CHANNEL_AXIS = 3


def _bn_relu(input):
    """Helper to build a BN -> relu block
    """
    norm = BatchNormalization(axis=CHANNEL_AXIS)(input)
    return Activation("relu")(norm)


def _conv_bn_relu(**conv_params):
    """Helper to build a conv -> BN -> relu block
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    def f(input):
        conv = Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(input)
        return _bn_relu(conv)

    return f


def _bn_relu_conv(**conv_params):
    """Helper to build a BN -> relu -> conv block.
    This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    def f(input):
        activation = _bn_relu(input)
        return Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(activation)

    return f


class BasicBlock(tf.keras.layers.Layer):

    def __init__(self, filter_num, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=(3, 3),
                                            strides=stride,
                                            padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=(3, 3),
                                            strides=1,
                                            padding="same")
        self.bn2 = tf.keras.layers.BatchNormalization()
        if stride != 1:
            self.downsample = tf.keras.Sequential()
            self.downsample.add(tf.keras.layers.Conv2D(filters=filter_num,
                                                       kernel_size=(1, 1),
                                                       strides=stride))
            self.downsample.add(tf.keras.layers.BatchNormalization())
        else:
            self.downsample = lambda x: x

    def call(self, inputs, training=None, **kwargs):
        residual = self.downsample(inputs)

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)

        output = tf.nn.relu(tf.keras.layers.add([residual, x]))

        return output


class BottleNeck(tf.keras.layers.Layer):
    def __init__(self, filter_num, stride=1):
        super(BottleNeck, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=(3, 3),
                                            strides=stride,
                                            padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv3 = tf.keras.layers.Conv2D(filters=filter_num * 4,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding='same')
        self.bn3 = tf.keras.layers.BatchNormalization()

        self.downsample = tf.keras.Sequential()
        self.downsample.add(tf.keras.layers.Conv2D(filters=filter_num * 4,
                                                   kernel_size=(1, 1),
                                                   strides=stride))
        self.downsample.add(tf.keras.layers.BatchNormalization())

    def call(self, inputs, training=None, **kwargs):
        residual = self.downsample(inputs)

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv3(x)
        x = self.bn3(x, training=training)

        output = tf.nn.relu(tf.keras.layers.add([residual, x]))

        return output


def make_basic_block_layer(filter_num, blocks, stride=1):
    res_block = tf.keras.Sequential()
    res_block.add(BasicBlock(filter_num, stride=stride))

    for _ in range(1, blocks):
        res_block.add(BasicBlock(filter_num, stride=1))

    return res_block


def make_bottleneck_layer(filter_num, blocks, stride=1):
    res_block = tf.keras.Sequential()
    res_block.add(BottleNeck(filter_num, stride=stride))

    for _ in range(1, blocks):
        res_block.add(BottleNeck(filter_num, stride=1))

    return res_block


import tensorflow as tf

# from config import NUM_CLASSES
# from models.residual_block import make_basic_block_layer, make_bottleneck_layer
NUM_CLASSES = 5


class ResNetTypeI(tf.keras.Model):
    def __init__(self, layer_params):
        super(ResNetTypeI, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(filters=64,
                                            kernel_size=(7, 7),
                                            strides=2,
                                            padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                               strides=2,
                                               padding="same")

        self.layer1 = make_basic_block_layer(filter_num=64,
                                             blocks=layer_params[0])
        self.layer2 = make_basic_block_layer(filter_num=128,
                                             blocks=layer_params[1],
                                             stride=2)
        self.layer3 = make_basic_block_layer(filter_num=256,
                                             blocks=layer_params[2],
                                             stride=2)
        self.layer4 = make_basic_block_layer(filter_num=512,
                                             blocks=layer_params[3],
                                             stride=2)

        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.layers.Dense(units=NUM_CLASSES, activation=tf.keras.activations.softmax)

    def build(self, training=None, mask=None,inputs=(512,512,3)):
        input = Input(shape=inputs)
        x = self.conv1(input)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool1(x)
        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)

        FTGenerator7 = _conv_bn_relu(filters=64, kernel_size=(3, 3), strides=(1, 1), name='FT6')(x)
        FTGenerator6 = _conv_bn_relu(filters=64, kernel_size=(3, 3), strides=(1, 1), name='FT5')(FTGenerator7)
        FTGenerator5 = _conv_bn_relu(filters=32, kernel_size=(3, 3), strides=(1, 1), name='FT4')(FTGenerator6)
        FTGenerator4 = _conv_bn_relu(filters=64, kernel_size=(3, 3), strides=(1, 1), name='FT3')(FTGenerator5)
        FTGenerator3 = _conv_bn_relu(filters=128, kernel_size=(3, 3), strides=(1, 1), name='FT2')(FTGenerator4)
        FTGenerator2 = UpSampling2D(size=(4, 4), name='FT2')(FTGenerator3)
        FTGenerator1 = UpSampling2D(size=(2, 2), name='FT1')(FTGenerator2)
        FTGenerator = _conv_bn_relu(filters=1, kernel_size=(3, 3), strides=(1, 1), name='FT')(FTGenerator1)

        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)
        x = self.avgpool(x)

        flatten1 = Flatten()(x)
        flatten1 = Dropout(0.4)(flatten1)
        D1_faceImg = Dense(40, activation='relu', name='D1_faceImg')(flatten1)

        D1_SpoofTypeLabel = Dense(128, activation='relu', name='D1_SpoofTypeLabel')(flatten1)
        D1_SpoofTypeLabel = Dropout(0.4)(D1_SpoofTypeLabel)
        D2_SpoofTypeLabel = Dense(11, activation='softmax', name='D2_SpoofTypeLabel')(D1_SpoofTypeLabel)

        D1_IlluminationLabel = Dense(128, activation='relu', name='D1_IlluminationLabel')(flatten1)
        D1_IlluminationLabel = Dropout(0.4)(D1_IlluminationLabel)
        D2_IlluminationLabel = Dense(5, activation='softmax', name='D2_IlluminationLabel')(D1_IlluminationLabel)

        D1_LiveLabel = Dense(128, activation='relu')(flatten1)
        D1_LiveLabel = Dropout(0.4)(D1_LiveLabel)
        D2_LiveLabel = Dense(2, activation='softmax', name='D2_LiveLabel')(D1_LiveLabel)

        model = Model(inputs=input,
                      outputs=[FTGenerator, D1_faceImg, D2_SpoofTypeLabel, D2_IlluminationLabel, D2_LiveLabel])
        return model


class ResNetTypeII(tf.keras.Model):
    def __init__(self, layer_params, inputs):
        super(ResNetTypeII, self).__init__()
        self.inputs = inputs
        self.conv1 = tf.keras.layers.Conv2D(filters=64,
                                            kernel_size=(7, 7),
                                            strides=2,
                                            padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                               strides=2,
                                               padding="same")

        self.layer1 = make_bottleneck_layer(filter_num=64,
                                            blocks=layer_params[0])
        self.layer2 = make_bottleneck_layer(filter_num=128,
                                            blocks=layer_params[1],
                                            stride=2)
        self.layer3 = make_bottleneck_layer(filter_num=256,
                                            blocks=layer_params[2],
                                            stride=2)
        self.layer4 = make_bottleneck_layer(filter_num=512,
                                            blocks=layer_params[3],
                                            stride=2)

        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()

    def build(self, training=None, mask=None):
        input = Input(shape=self.inputs)
        x = self.conv1(input)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool1(x)
        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)

        FTGenerator7 = _conv_bn_relu(filters=64, kernel_size=(3, 3), strides=(1, 1), name='FT6')(x)
        FTGenerator6 = _conv_bn_relu(filters=64, kernel_size=(3, 3), strides=(1, 1), name='FT5')(FTGenerator7)
        FTGenerator5 = _conv_bn_relu(filters=32, kernel_size=(3, 3), strides=(1, 1), name='FT4')(FTGenerator6)
        FTGenerator4 = _conv_bn_relu(filters=64, kernel_size=(3, 3), strides=(1, 1), name='FT3')(FTGenerator5)
        FTGenerator3= _conv_bn_relu(filters=128, kernel_size=(3, 3), strides=(1, 1), name='FT2')(FTGenerator4)
        FTGenerator2 = UpSampling2D(size=(4, 4), name='FT2')(FTGenerator3)
        FTGenerator1 = UpSampling2D(size=(2, 2), name='FT1')(FTGenerator2)
        FTGenerator = _conv_bn_relu(filters=1, kernel_size=(3, 3), strides=(1, 1), name='FT')(FTGenerator1)

        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)
        x = self.avgpool(x)

        flatten1 = Flatten()(x)
        flatten1 = Dropout(0.4)(flatten1)
        FaceAttributeLabel = Dense(40, activation='relu', name='D1_faceImg')(flatten1)

        D1_SpoofTypeLabel = Dense(128, activation='relu', name='D1_SpoofTypeLabel')(flatten1)
        D1_SpoofTypeLabel = Dropout(0.4)(D1_SpoofTypeLabel)
        D2_SpoofTypeLabel = Dense(11, activation='softmax', name='D2_SpoofTypeLabel')(D1_SpoofTypeLabel)

        D1_IlluminationLabel = Dense(128, activation='relu', name='D1_IlluminationLabel')(flatten1)
        D1_IlluminationLabel = Dropout(0.4)(D1_IlluminationLabel)
        D2_IlluminationLabel = Dense(5, activation='softmax', name='D2_IlluminationLabel')(D1_IlluminationLabel)

        D1_LiveLabel = Dense(128, activation='relu')(flatten1)
        D1_LiveLabel = Dropout(0.4)(D1_LiveLabel)
        D2_LiveLabel = Dense(2, activation='softmax', name='D2_LiveLabel')(D1_LiveLabel)

        model = Model(inputs=input,
                      outputs=[FTGenerator, FaceAttributeLabel, D2_SpoofTypeLabel, D2_IlluminationLabel, D2_LiveLabel])
        return model


def resnet_18():
    model = ResNetTypeI(layer_params=[2, 2, 2, 2])
    return model.build(inputs=(256, 256, 3))


def resnet_34():
    model = ResNetTypeI(layer_params=[3, 4, 6, 3])
    return model.build(inputs=(256, 256, 3))


def resnet_50():
    model = ResNetTypeII(layer_params=[3, 4, 6, 3], inputs=(256, 256, 3))
    return model.build()


def resnet_101():
    model = ResNetTypeII(layer_params=[3, 4, 23, 3], inputs=(256, 256, 3))
    return model.build()


def resnet_152():
    model = ResNetTypeII(layer_params=[3, 8, 36, 3], inputs=(512, 512, 3))
    return model.build()


if __name__ == "__main__":
    model = resnet_50()
    layer_output = model.get_layer("conv2d_70").output
    model.summary()