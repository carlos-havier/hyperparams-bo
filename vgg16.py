import argparse

from keras.layers import (
    Conv2D, BatchNormalization, Activation,
    MaxPooling2D, Dense, Flatten
)

from model import BaseModel
from utils import load_mnist
import hypers


def vgg(input_tensor):
    """Inference function for VGGNet

    y = vgg(X)

    Parameters
    ----------
    input_tensor : keras.layers.Input

    Returns
    ----------
    y : softmax output tensor
    """
    def two_conv_pool(x, F1, F2, name):
        x = Conv2D(F1, (3, 3), activation=None, padding='same', name='{}_conv1'.format(name))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(F2, (3, 3), activation=None, padding='same', name='{}_conv2'.format(name))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='{}_pool'.format(name))(x)

        return x

    def three_conv_pool(x, F1, F2, F3, name):
        x = Conv2D(F1, (3, 3), activation=None, padding='same', name='{}_conv1'.format(name))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(F2, (3, 3), activation=None, padding='same', name='{}_conv2'.format(name))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(F3, (3, 3), activation=None, padding='same', name='{}_conv3'.format(name))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='{}_pool'.format(name))(x)

        return x

    net = input_tensor

    net = two_conv_pool(net, 64, 64, "block1")
    net = two_conv_pool(net, 128, 128, "block2")
    net = three_conv_pool(net, 256, 256, 256, "block3")
    net = three_conv_pool(net, 512, 512, 512, "block4")

    net = Flatten()(net)
    net = Dense(512, activation='relu', name='fc')(net)
    net = Dense(10, activation='softmax', name='predictions')(net)

    return net


class VGGNet(BaseModel):
    def __init__(self, model_path):
        super(VGGNet, self).__init__("VGG", vgg, model_path)


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("epoch", default=str(hypers.hyperparameters['epochs']), type=int, help="Epochs")
    parser.add_argument("--model_path", default="model/vggnet.h5", type=str, help="model path (default: model/vggnet.h5)")

    args = parser.parse_args()
    return args.epoch, args.model_path

def i_main(EPOCH, MODEL_PATH):

    # (X, y)
    train, valid, _ = load_mnist(samplewise_normalize=True)

    vggnet = VGGNet(MODEL_PATH)
    vggnet.fit((train[0], train[1]), (valid[0], valid[1]), EPOCH)

def main():
    EPOCH, MODEL_PATH = arg_parser()

    # (X, y)
    train, valid, _ = load_mnist(samplewise_normalize=True)

    vggnet = VGGNet(MODEL_PATH)
    vggnet.fit((train[0], train[1]), (valid[0], valid[1]), EPOCH)


if __name__ == '__main__':
    main()
