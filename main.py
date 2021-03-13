import dataset as mdataset
import model as mmodel


def main():
    dataset, num_samples, batch_size = mdataset.get_dataset_TFRecord()
    print("Training on {} images".format(num_samples))

    steps_per_epoch = num_samples // batch_size
    model = mmodel.build()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[
        'accuracy'])  # if you have one-hot encoded your target in order to have 2D shape (n_samples, n_class), you can use categorical_crossentropy
    # if you have 1D integer encoded target, you can use sparse_categorical_crossentropy as loss function
    mmodel.train(model, dataset, steps_per_epoch)

if __name__ == '__main__':
    main()