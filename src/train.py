import datetime
from model import *
import tensorflow as tf
from prepare_data import Data_generator
from tensorflow.keras.callbacks import ModelCheckpoint


data_train = Data_generator(8, "train")
data_val = Data_generator(8,"test")
model = resnet_18()
model.summary()
losses = ["mse","binary_crossentropy",
          "categorical_crossentropy","categorical_crossentropy",
          "categorical_crossentropy"]
filepath = "../weight/saved-model-{epoch:02d}-{val_acc:.2f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, mode='max')
log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
lossWeights = [0.1, 1.0, 0.1, 0.01, 1.0]
model.compile(optimizer="adam", loss=losses, loss_weights=lossWeights,
              metrics=["accuracy"])
model.fit_generator(generator= data_train, steps_per_epoch=int(230000/8),epochs = 10,
                    verbose= 1,validation_data=data_val,validation_steps=int(25000/8),
                    callbacks=[tensorboard_callback, checkpoint] )