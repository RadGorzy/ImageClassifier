import dataset as mdataset
import model as mmodel

import tensorflow as tf
import numpy as np

def main():
    train_dataset, test_dataset=mdataset.get_dataset()
    train_dataset,test_dataset=mdataset.prepare(train_dataset,test_dataset)
    """
    BATCH_SIZE = 64
    SHUFFLE_BUFFER_SIZE = 100

    train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    test_dataset = test_dataset.batch(BATCH_SIZE)
"""


    model=mmodel.build()
    mmodel.train(model,train_dataset)
def test():
    DATA_URL = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'

    path = tf.keras.utils.get_file('mnist.npz', DATA_URL)
    with np.load(path) as data:
        train_examples = data['x_train']
        train_labels = data['y_train']
        test_examples = data['x_test']
        test_labels = data['y_test']


    train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_examples, test_labels))

    BATCH_SIZE = 64
    SHUFFLE_BUFFER_SIZE = 100

    train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    test_dataset = test_dataset.batch(BATCH_SIZE)

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

    model.compile(optimizer=tf.keras.optimizers.RMSprop(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['sparse_categorical_accuracy'])

    model.fit(train_dataset, epochs=10)

    model.evaluate(test_dataset)
def tfrecordTest():
    dataset,num_samples,batch_size=mdataset.get_dataset_TFRecord()
    steps_per_epoch=num_samples // batch_size
    model = mmodel.build_v1()
    mmodel.trainSteps(model,dataset,steps_per_epoch)
if __name__ == '__main__':
    main()
    #test()
    #tfrecordTest()