'''
This will contain the actual neural network,
This will teach itself how to drive ...
    what does this mean?
    This means a couple things
1. we WONT HARD CODE any rules into the car, it will learn the rules with our help (supervised learning)
2.
'''

import cv2      # for NN

# LEARN HOW TO MAKE A NEURAL NETWORK AND TRAIN IT

class Tesla(object):
    """docstring for Tesla."""

    def __init__(self):
        super(Tesla, self).__init__()
        # self.arg = arg
        self.posx = 0
        self.posy = 0
        # kind of like x, y coordinates
        # x  = 0 STRAIGHT, -2 -1 left, 1 2 right
        # y  = 0 STOP -2 -1 reverse, 1 2 forwards,
        self.current_direction = (0, 0)

    # return as a string
    def __str__(self):
        return 'Dir='+str(self.current_direction)

    def get_pos(self):
        return (self.posx, self.posy)

    def get_direction(self):
        return self.current_direction

    def set_pos(self, new_x, new_y):
        self.posx = new_x
        self.posy = new_y

    def set_direction(self, newx, newy):
        self.current_direction = newx, newy
        # send instructions to arduino

    # CAN I MAKE A FUNCTION THAT CAN DETECT TURNS?

    def __nn__(self):
        pass
        # for future alan, the one who knows how to make a nn and train it

#
# import keras
# from keras.datasets import mnist
# from keras.utils import to_categorical
# from keras.models import Sequential
# from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
# from keras.preprocessing.image import ImageDataGenerator
#
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
#
# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
# x_train /= 255  # colors?
# x_test /= 255   # colors?
#
# print(x_train[0].shape)
# print(x_test[0].shape)
# # x_train.shape[0]
# x_train = x_train.reshape(60000, 28, 28, 1)
# x_test = x_test.reshape(10000, 28, 28, 1)
# # x_test.shape[0]
#
# print(x_train)
# print(x_test)
# y_train = to_categorical(y_train, 10)
# y_test = to_categorical(y_test, 10)
#
# model = Sequential() # easier one to use
# # model = keras.models.load_model("model.h5")
# model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu',
#     input_shape=x_train.shape[1:]))
# model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
# model.add(MaxPool2D(pool_size=(2, 2)))
# model.add(Dropout(rate=0.25))
# model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
# model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
# model.add(MaxPool2D(pool_size=(2, 2)))
# model.add(Dropout(rate=0.25))
# model.add(Flatten()) # tf
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(rate=0.5))
# model.add(Dense(10, activation='softmax'))
#
# model.compile(
#     loss='categorical_crossentropy',
#     optimizer='adam',
#     metrics=['accuracy']
# )
#
# model.fit(x_train[:20000], y_train[:20000],
#     validation_data=(x_test, y_test), epochs=3)
#
# model.save("model.h5")
