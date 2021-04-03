import dataset as mdataset
import model as mmodel

import tensorflow as tf
import datetime


def main():
    dataset, num_samples, batch_size = mdataset.get_dataset_TFRecord()
    print("Training on {} images".format(num_samples))

    steps_per_epoch = num_samples // batch_size
    model = mmodel.build()

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    mmodel.train(model, dataset, steps_per_epoch,[tensorboard_callback])

if __name__ == '__main__':
    main()