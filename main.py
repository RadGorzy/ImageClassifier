import dataset as mdataset
import model as mmodel

import tensorflow as tf
import datetime

_BATCH_SIZE = 100
_EPOCHS = 5
def main():
    trainDataset, trainNumSamples = mdataset.get_dataset_TFRecord(['/home/radek/Projects/ImageClassifier/data/TFRecords/train/train-00000-of-00004.tfrecord'],_BATCH_SIZE)
    print("Training on {} images".format(trainNumSamples))

    validDataset, validNumSamples = mdataset.get_dataset_TFRecord(['/home/radek/Projects/ImageClassifier/data/TFRecords/test/test-00000-of-00004.tfrecord'],_BATCH_SIZE)
    print("Validating on {} images".format(validNumSamples))

    trainStepsPerEpoch = trainNumSamples // _BATCH_SIZE
    validationSteps = validNumSamples // _BATCH_SIZE

    model = mmodel.build()

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    mmodel.train(model, trainDataset, _EPOCHS, trainStepsPerEpoch,validDataset,validationSteps,[tensorboard_callback])

if __name__ == '__main__':
    main()