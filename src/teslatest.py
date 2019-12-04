import sys
from keras.models import load_model
from keras.datasets import mnist
from numpy import loadtxt
from keras.datasets import mnist
from keras.utils import to_categorical


(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_test = x_test.astype('float32')
x_test /= 255   # colors?

# print(x_train[0].shape)
print(x_test[0].shape)
# x_train.shape[0]
# x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)
# x_test.shape[0]

# print(x_train)
print(x_test)
# y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)


def main(arg):
    print("Starting teslatest.py ... ")
    print(arg)
    # file = h5py.File("model.h5", 'r')
    model = load_model(file)
    # model.summary()

    # dataset = loadtxt(mnist)
    # dataset = loadtxt(mnist.load_data())
    scores = model.evaluate(x_test, y_test, verbose=0)
    print("accuracy:", (scores[1]*100))

if __name__ == '__main__':
    file = sys.argv[1]
    main(file)
