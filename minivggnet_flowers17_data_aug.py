__author__ = 'kevin'
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from datasets.simpledataloader import SimpleDataLoader
from preprocessing.aspectawarepreprocessor import AspectAwarePreprocessor
from preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from preprocessing.simplepreprocessor import SimplePreprocessor
from nn.conv.minivvgnet import MiniVGGNet
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", help="path to input dataset", required=True)
args = vars(ap.parse_args())

print("[INFO] loading images...")
imagesPaths = list(paths.list_images(args["dataset"]))
className = [pt.split(os.path.sep)[-2] for pt in imagesPaths]
className = [str(x) for x in np.unique(className)]

print("className = {}".format(len(className)))

aap = AspectAwarePreprocessor(64, 64)
iap = ImageToArrayPreprocessor()

sdl = SimpleDataLoader(preprocessor=[aap, iap])
(data, labels) = sdl.load(imagesPaths, verbose=500)

data = data.astype("float") / 255.0

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)
print("Sizes --- trainX = {} testX = {} trainY = {} testY={}".format(len(trainX), len(testX), len(trainY), len(testY)))
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

print("[INFO] compiling model...")
opt = SGD(lr=0.05)
model = MiniVGGNet.built(width=64, height=64, depth=3, classes=len(className))
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

print("[INFO] training network...")

H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=32, epochs=100, verbose=1)

print("[INFO] Evaluating  network...")

predictions = model.predict(testX, batch_size=32)

print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=className))

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 100), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, 100), H.history["val_acc"], label="val_acc")

plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
