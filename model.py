import dataset
import tensorflow as tf

def build():

    from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, Input
    inputs = Input(shape=(299, 299, 1))
    x = Conv2D(32, (3, 3), activation='relu', padding='valid')(inputs)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(5, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)

    return model

def train(model,dataset,stepsPerEpoch):
    model.fit(dataset,epochs=1,steps_per_epoch=stepsPerEpoch)