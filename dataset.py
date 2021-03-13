import numpy as np
import tensorflow as tf

def get_features(example_proto):
    features = {
        "image/encoded": tf.io.FixedLenFeature([], tf.string),
        "image/height": tf.io.FixedLenFeature([], tf.int64),
        "image/width": tf.io.FixedLenFeature([], tf.int64),
        "image/filename": tf.io.FixedLenFeature([], tf.string),
        "image/class/label": tf.io.FixedLenFeature([], tf.int64), }
    parsed_features = tf.io.parse_single_example(example_proto, features)
    # print("Features" + str(parsed_features))
    return parsed_features["image/encoded"], parsed_features["image/height"], parsed_features["image/width"],parsed_features["image/filename"], parsed_features["image/class/label"]  #parsed_features will return this parameters in random order
def edit(feature, label,resize=0):
    image = tf.image.decode_jpeg(feature, channels=1)  # image_raw
    if resize == 1:
        image = tf.image.resize_image_with_crop_or_pad(image, 299,
                                                       299)  # cropped image with size 299x299
    image = tf.multiply(tf.cast(image, tf.float32), 1.0 / 255.0) #from uint8 to float32 and from 0-255 do 0-1

    label = tf.one_hot(label, depth=5)
    return image,label
def get_futures_and_edit(example_proto):
    feature,_,_,_,label=get_features(example_proto)
    return edit(feature,label,resize=0)

def get_dataset_TFRecord():
    batch_size=100
    tfrecordList=['/home/radek/Projects/ImageClassifier/TFRecords/test/test-00000-of-00004.tfrecord']
    """,
                   '/home/radek/Projects/ImageClassifier/TFRecords/test/test-00001-of-00004.tfrecord',
                   '/home/radek/Projects/ImageClassifier/TFRecords/test/test-00002-of-00004.tfrecord',
                   '/home/radek/Projects/ImageClassifier/TFRecords/test/test-00003-of-00004.tfrecord']"""
    dataset = tf.data.TFRecordDataset(filenames=[tfrecordList])
    num_samples=0
    for el in dataset:
        num_samples+=1


    dataset=dataset.map(get_futures_and_edit,num_parallel_calls=8)
    # only shuffle if shuffle flag
    dataset = dataset.shuffle(10000)

    # take only dataset of length batch_size
    dataset = dataset.batch(batch_size)

    # make sure you can repeatedly take datasets from the TFRecord
    dataset = dataset.repeat()

    # Return the dataset.
    return dataset, num_samples, batch_size
    """
    datasetLen=0
    for el in dataset:
        datasetLen+=1
    print(datasetLen)
    """
    """
    for el in dataset.take(1):
        #print(repr(el))
        example = tf.train.Example()
        example.ParseFromString(el.numpy())
        print(example)
    """
def get_dataset_TFRecord_test():
    tfrecordList = ['/home/radek/Projects/ImageClassifier/TFRecords/test/test-00000-of-00004.tfrecord']
    """,
                   '/home/radek/Projects/ImageClassifier/TFRecords/test/test-00001-of-00004.tfrecord',
                   '/home/radek/Projects/ImageClassifier/TFRecords/test/test-00002-of-00004.tfrecord',
                   '/home/radek/Projects/ImageClassifier/TFRecords/test/test-00003-of-00004.tfrecord']"""
    dataset = tf.data.TFRecordDataset(filenames=[tfrecordList])
    for el in dataset.take(1):
        print(repr(el))
        image,label=get_futures_and_edit(el)
        print("LABEL= "+repr(label))


if __name__=='__main__':
    #get_dataset_TFRecord()
    #get_dataset()
    get_dataset_TFRecord_test()

