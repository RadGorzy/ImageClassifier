import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

def build():


    inputs = Input(shape=(299, 299,1))
    print(repr(inputs))
    x = Conv2D(6, (6, 6), activation='relu', padding='same')(inputs)
    print(repr(x))
    #x = MaxPool2D(pool_size=(2, 2))(x)
    x = Conv2D(12, (5, 5),strides=(2,2), activation='relu', padding='same')(x)
    print(repr(x))
    #x = MaxPool2D(pool_size=(2, 2))(x)
    x = Conv2D(24, (4, 4),strides=(2,2), activation='relu', padding='same')(x)
    print(repr(x))
    x = Conv2D(48, (3, 3), strides=(2, 2), activation='relu', padding='same')(x)
    print(repr(x))
    x = Conv2D(96, (3, 3), strides=(2, 2), activation='relu', padding='same')(x)
    print(repr(x))
    x = Flatten()(x)
    print(repr(x))
    x = Dense(200, activation='relu')(x)
    print(repr(x))
    x = Dropout(0.25)(x)
    outputs = Dense(5, activation='softmax')(x)
    print(repr(outputs))

    model = tf.keras.Model(inputs, outputs)


    """
    from tensorflow.keras.models import Sequential
    model = Sequential()
    model.add(Conv2D(6, (6, 6), activation='relu', padding='same',
                     input_shape=(299, 299, 1)))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(12, (5, 5),strides=(2,2), activation='relu', padding='same'))
    model.add(Conv2D(24, (4, 4),strides=(2,2), activation='relu', padding='same'))
    model.add(Conv2D(48, (3, 3), strides=(2, 2), activation='relu', padding='same'))
    model.add(Conv2D(96, (3, 3), strides=(2, 2), activation='relu', padding='same'))
    model.add(Flatten())
    model.add(Dense(200, activation='relu'))
    model.add(Dense(5, activation='softmax'))
    """
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.Reshape(input_shape=(299 * 299,), target_shape=(299, 299, 1)),
        tf.keras.layers.Conv2D(kernel_size=3,filters=12, activation='relu', padding='same'),
        tf.keras.layers.Conv2D(kernel_size=6, filters=24, activation='relu', padding='same', strides=2),
        tf.keras.layers.Conv2D(kernel_size=6, filters=32, activation='relu', padding='same', strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(200, activation='relu'),
        tf.keras.layers.Dense(5, activation='softmax')
    ])
    """
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.Reshape(input_shape=(299 * 299,), target_shape=(299, 299, 1)),
        tf.keras.layers.Conv2D(kernel_size=6, filters=6, activation='relu', padding='same'),
        tf.keras.layers.Conv2D(kernel_size=5, filters=12, activation='relu', padding='same', strides=2),
        tf.keras.layers.Conv2D(kernel_size=4, filters=24, activation='relu', padding='same', strides=2),
        tf.keras.layers.Conv2D(kernel_size=3, filters=48, activation='relu', padding='same', strides=2),
        tf.keras.layers.Conv2D(kernel_size=3, filters=96, activation='relu', padding='same', strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(200, activation='relu'),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(5, activation='softmax')
    ])
    """
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=Adam(lr=0.001),
    metrics=[tf.keras.metrics.CategoricalAccuracy()])  # if you have one-hot encoded your target in order to have 2D shape (n_samples, n_class), you can use categorical_crossentropy
# if you have 1D integer encoded target, you can use sparse_categorical_crossentropy as loss function

    return model

def train(model,dataset,stepsPerEpoch,callback):
    model.fit(dataset,epochs=1,steps_per_epoch=stepsPerEpoch,callbacks=callback)