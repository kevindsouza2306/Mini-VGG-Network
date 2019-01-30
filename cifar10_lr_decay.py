__author__ = 'kevin'

import matplotlib

matplotlib.use("Agg")

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.datasets import cifar10
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler
from nn.conv.minivvgnet import MiniVGGNet
import matplotlib.pyplot as plt
import argparse
import numpy as np


def step_decay(epoch):
    initAlfa = 0.01
    factor = 0.25
    dropEvery = 5

    alpha = initAlfa * (factor ** np.floor((1 + epoch) / dropEvery))

    return float(alpha)


ap = argparse.ArgumentParser()

ap.add_argument("-o", "--output", required=True, help="Path to the output Loss/Accuracy plot")

args = vars(ap.parse_args())

print("[INFO] loading CIFAR-10 data...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

# initialize the label names for the CIFAR-10 dataset
labelNames = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

callback = [LearningRateScheduler(step_decay)]

opt = SGD(lr=0.01, nesterov=True, momentum=0.9)

model = MiniVGGNet.built(width=32, height=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=64, epochs=40, callbacks=callback, verbose=1)

model.save("deacay_model.h5")
print("[INFO] evaluating network...")
prediction = model.predict(testX, batch_size=64)
print(classification_report(testY.argmax(axis=1), prediction.argmax(axis=1), target_names=labelNames))

# plotting the loss accuracy graph
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 40), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 40), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 40), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, 40), H.history["val_acc"], label="val_acc")

plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
