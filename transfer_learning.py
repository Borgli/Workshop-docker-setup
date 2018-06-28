import datetime
import os
import subprocess

from keras import Model
from keras.applications import VGG16
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, CSVLogger
from keras.layers import GlobalAveragePooling2D, Dense
from keras.preprocessing.image import ImageDataGenerator


def __create_image_generators(dataset_path, image_size_x, image_size_y, batch_size):
    # Read Data and Augment it: Make sure to select augmentations that are appropriate to your images.
    # To save augmentations un-comment save lines and add to your flow parameters.
    def get_samples(dataset_dir):
        num_samples = 0
        for label in os.listdir(dataset_dir):
            num_samples += len(os.listdir(os.path.join(dataset_dir, label)))
        return num_samples

    train_samples = get_samples(os.path.abspath(os.path.join(dataset_path, "train")))

    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       # rotation_range=transformation_ratio,
                                       # shear_range=transformation_ratio,
                                       # zoom_range=transformation_ratio,
                                       # cval=transformation_ratio,
                                       horizontal_flip=True,
                                       vertical_flip=True)

    test_samples = get_samples(os.path.abspath(os.path.join(dataset_path, "test")))

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(os.path.abspath(os.path.join(dataset_path, "train")),
                                                        target_size=(image_size_x, image_size_y),
                                                        batch_size=batch_size,
                                                        class_mode='categorical')


    # save_to_dir=os.path.join(os.path.abspath(train_data_dir), '../preview')
    # save_prefix='aug',
    # save_format='jpeg')
    # use the above 3 commented lines if you want to save and look at how the data augmentations look like

    test_generator = test_datagen.flow_from_directory(os.path.abspath(os.path.join(dataset_path, "test")),
                                                      target_size=(image_size_x, image_size_y),
                                                      batch_size=batch_size,
                                                      class_mode='categorical')

    return train_generator, test_generator, train_samples, test_samples


def __create_model(image_size_x, image_size_y, num_classes):

    base_model = VGG16(input_shape=(image_size_x, image_size_y, 3),
                             weights='imagenet', include_top=False)

    # Top Model Block
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    # add your top layer block to your base model
    model = Model(base_model.input, predictions)
    # print(model.summary())

    # # let's visualize layer names and layer indices to see how many layers/blocks to re-train
    # # uncomment when choosing based_model_last_block_layer
    # for i, layer in enumerate(model.layers):
    #     print(i, layer.name)
    return model, base_model


def train_classification_block(model, base_model, train_generator, test_generator, batch_size, train_samples, test_samples, log_dir, best_model_path):
    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all layers of the based model that is already pre-trained.
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer="nadam", loss='categorical_crossentropy', metrics=['accuracy'])

    callbacks_list = [
        ModelCheckpoint(best_model_path, monitor='val_acc', verbose=1, save_best_only=True),
        EarlyStopping(monitor='val_acc', patience=5, verbose=0),
        TensorBoard(log_dir=log_dir, histogram_freq=0, batch_size=batch_size,
                    write_graph=True, write_grads=False, write_images=True),
        CSVLogger(filename=os.path.abspath(os.path.join(log_dir, "epoch_results.csv")))
    ]

    print("\nStarting to train simple CNN\n")
    # Train Simple CNN
    history = model.fit_generator(train_generator,
                                  steps_per_epoch=100,
                                  epochs=5,
                                  validation_data=test_generator,
                                  validation_steps=100,
                                  callbacks=callbacks_list)


def train_fine_tuning(model, delimiting_layer, train_generator, test_generator, batch_size, train_sample, test_sample, log_dir, best_model_path):
    print("\nStarting to Fine Tune Model\n")

    # add the best weights from the train top model
    # at this point we have the pre-train weights of the base model and the trained weight of the new/added top model
    # we re-load model weights to ensure the best epoch is selected and not the last one.
    model.load_weights(best_model_path)

    # based_model_last_block_layer_number points to the layer in your model you want to train.
    # For example if you want to train the last block of a 19 layer VGG16 model this should be 15
    # If you want to train the last Two blocks of an Inception model it should be 172
    # layers before this number will used the pre-trained weights, layers above and including this number
    # will be re-trained based on the new data.

    for layer in model.layers[:delimiting_layer]:
        layer.trainable = False
    for layer in model.layers[delimiting_layer:]:
        layer.trainable = True


    # compile the model with a SGD/momentum optimizer
    # and a very slow learning rate.
    model.compile(optimizer="nadam", loss='categorical_crossentropy', metrics=['accuracy'])

    # save weights of best training epoch: monitor either val_loss or val_acc

    callbacks_list = [
        ModelCheckpoint(os.path.join(log_dir, "best_final_model.h5py"), monitor='val_acc', verbose=1, save_best_only=True),
        EarlyStopping(monitor='val_acc', patience=5, verbose=1),
        TensorBoard(log_dir=os.path.join(log_dir, 'tune'), histogram_freq=0, batch_size=batch_size,
                    write_graph=True, write_grads=False, write_images=True),
        CSVLogger(filename=os.path.join(log_dir, "epoch_results_tune.csv"))
    ]

    # fine-tune the model
    history = model.fit_generator(train_generator, steps_per_epoch=100, epochs=5, validation_data=test_generator,
                                  validation_steps=100, callbacks=callbacks_list)

    return max(history.history['val_acc'])


def __train_model(train_generator, test_generator, model, base_model, batch_size, train_samples, test_samples, log_dir):
    if not os.path.exists(os.path.abspath(log_dir)):
        os.mkdir(os.path.abspath(log_dir))

    best_model_path = os.path.join(log_dir, "best_model.h5py")
    train_classification_block(model, base_model, train_generator, test_generator, batch_size,
                               train_samples, test_samples, log_dir, best_model_path)
    train_fine_tuning(model, 16, train_generator, test_generator, batch_size,
                                 train_samples, test_samples, log_dir, best_model_path)


def train_network(dataset_path):
    image_size_x, image_size_y, batch_size, num_classes = 48, 48, 32, 4

    train_generator, test_generator, train_samples, test_samples = __create_image_generators(dataset_path, image_size_x,
                                                                                             image_size_y, batch_size)
    model, base_model = __create_model(image_size_x, image_size_y, num_classes)
    __train_model(train_generator, test_generator, model, base_model, batch_size, train_samples, test_samples,
                  os.path.join("logs", "log-{}".format(datetime.datetime.now()).replace(" ", "_")))


tensorboard = None
try:
    tensorboard = subprocess.Popen(["tensorboard", "--host localhost", "--logdir=log"])
    train_network("splitted-dataset")
    tensorboard.wait()
finally:
    if tensorboard:
        tensorboard.kill()
