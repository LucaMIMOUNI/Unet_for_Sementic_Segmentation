import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import skimage.io
import glob                                           #filename pattern matching
from modified_unet import Unet
from tensorflow.keras import layers

def get_images(path, output_shape=(None, None)):
    '''
    Loads images from path/{id}/images/{id}.png into a numpy array
    '''
    img_paths = ['{0}/{1}/images/{1}.png'.format(path, id) for id in os.listdir(path) if id != ".DS_Store"]
    X_data = np.array([skimage.io.imread(path)[:,:,:3] for path in img_paths], dtype="object")  #take only 3 channels/bands
    X_data = np.array([skimage.transform.resize(skimage.io.imread(path)[:,:,:3], output_shape=output_shape, mode='constant', preserve_range=True) for path in img_paths], dtype=np.uint8)  #take only 3 channels/bands

    return X_data

def get_labels(path, output_shape=(None, None)):
    '''
    Loads and concatenates images from path/{id}/masks/{id}.png into a numpy array
    '''
    img_paths = [glob.glob('{0}/{1}/masks/*.png'.format(path, id)) for id in os.listdir(path) if id != ".DS_Store"]

    Y_data = []
    for i, img_masks in enumerate(img_paths):  #loop through each individual nuclei for an image and combine them together
        masks = skimage.io.imread_collection(img_masks).concatenate()  #masks.shape = (num_masks, img_height, img_width)
        mask = np.max(masks, axis=0)                                   #mask.shape = (img_height, img_width)
        mask = skimage.transform.resize(mask, output_shape=output_shape+(1,), mode='constant', preserve_range=True)  #need to add an extra dimension so mask.shape = (img_height, img_width, 1)
        Y_data.append(mask)

    # make sure to return the binary images but coded as floats to be compatible with the loss functions
    Y_data = np.array(Y_data, dtype=np.bool_)

    return Y_data

# Importing the dataset
def get_dataset(dir, im_shape=(64, 64), do_visu=True):
    train_path = os.path.join(dir, 'stage1_train/stage1_train')
    test_path = os.path.join(dir, 'stage1_test/stage1_test')

    X_train = get_images(train_path, output_shape=im_shape)
    X_test = get_images(test_path, output_shape=im_shape)

    y_train = get_labels(train_path, output_shape=im_shape)
    y_test = None

    if do_visu:
        print('X_train:', X_train.shape)
        print('y_train:', y_train.shape)
        print('X_test:', X_test.shape)
        display_images(train_path)

    return X_train, y_train, X_test, y_test

def display_images(path, num_images=5):
    img_paths = ['{0}/{1}/images/{1}.png'.format(path, id) for id in os.listdir(path) if id != ".DS_Store"]
    plt.figure(figsize=(12, 6))
    for i in range(num_images):
        img = skimage.io.imread(img_paths[i])
        plt.subplot(1, num_images, i + 1)
        plt.imshow(img)
        plt.title(f'Image {i+1}')
        plt.axis('off')
    plt.show()

def display_masks(X_train, y_train, num_images=5):
    idx = np.random.randint(y_train.shape[0], size=1)[0]
    print ('Looking at image ', idx)
    plt.figure(figsize=(12, 6))
    plt.subplot(1,2,1)
    plt.imshow(skimage.exposure.rescale_intensity(X_train[idx, :, :, :]))
    plt.title('input image')
    plt.subplot(1,2,2)
    plt.imshow(y_train[idx, :, :],cmap='gray')
    plt.title('mask')
    plt.show()

def dice_coefficient(y_true, y_pred):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + 1)

# def data_augmentation(X_train, y_train, batch_size):
#     data_gen_args = dict(rotation_range=20,
#                          width_shift_range=0.2,
#                          height_shift_range=0.2,
#                          zoom_range=0.2,
#                          horizontal_flip=True,
#                          fill_mode='nearest')

#     image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**data_gen_args)
#     mask_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**data_gen_args)

#     seed = 1
#     image_datagen.fit(X_train, augment=True, seed=seed)
#     mask_datagen.fit(y_train, augment=True, seed=seed)

#     image_generator = image_datagen.flow(X_train, batch_size=batch_size, seed=seed)
#     mask_generator = mask_datagen.flow(y_train, batch_size=batch_size, seed=seed)

#     train_generator = zip(image_generator, mask_generator)
    
#     # Generate augmented data
#     X_train_augmented, y_train_augmented = next(train_generator)
    
#     return X_train_augmented, y_train_augmented, train_generator

def display_prediction(model, X_test):
    # if required reload the model
    model.load_weights('weights/model1_weights.weights.h5')
    predictions = model.predict(X_test)

    # Visualize some predictions
    num_images = 5
    plt.figure(figsize=(15, 15))
    for i in range(num_images):
        plt.subplot(3, num_images, i + 1)
        plt.imshow(X_test[i])
        plt.title(f'Test Image {i+1}')
        plt.axis('off')

        plt.subplot(3, num_images, i + 1 + num_images)
        plt.imshow(predictions[i].squeeze(), cmap='gray')
        plt.title(f'Predicted Mask {i+1}')
        plt.axis('off')

        plt.subplot(3, num_images, i + 1 + 2 * num_images)
        plt.imshow(predictions[i].squeeze(), cmap='gray')
        plt.imshow(X_test[i], alpha=0.5)
        plt.title(f'Overlay {i+1}')
        plt.axis('off')

    plt.show()


def main():
    ########
    # Init #
    ########
    IMDIR = './mini-input'
    im_height, im_width =64, 64
    epochs = 10
    batch_size = 32

    ###############
    # Get dataset #
    ###############
    X_train, y_train, X_test, _ = get_dataset(IMDIR, im_shape=(im_height,im_width), do_visu=False)
    y_train =1.0*y_train.astype(float)
    print(f"Training set data shape: {X_train.shape} with type: {X_train.dtype}")
    print(f"Training set label shape: {y_train.shape} with type: {y_train.dtype}")

    #############
    # Visualize #
    #############
    visu_mask = False
    if visu_mask:
        display_masks(X_train, y_train)

    ###############
    # Build Model #
    ###############
    model = Unet(input_size=(im_height, im_width, 3), classes=1, dropout=0.2)
    # model.summary()

    #####################
    # Data augmentation #
    #####################
    data_augmented = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomContrast(0.2),
        layers.RandomTranslation(0.2, 0.2)
    ])
    X_train = data_augmented(X_train)
    y_train = data_augmented(y_train)
    ###############
    # Train Model #
    ###############
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[dice_coefficient])
    # history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    # model.save_weights('model1_weights.weights.h5') # save the wieghts

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[dice_coefficient])
    history = model.fit(X_train, y_train, epochs=epochs, steps_per_epoch=len(X_train) // batch_size)
    model.save_weights('model1_weights_augmented.weights.h5') # save the weights

    ##############################
    # Visualize loss and metrics #
    ##############################
    visu_loss = False
    if visu_loss:
        plt.figure()
        plt.plot(history.history['loss'], label='Loss')
        plt.plot(history.history['accuracy'], label='accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.title('Training Loss and Dice Coefficient')
        plt.legend()
        plt.show()

    ##############
    # Prediction #
    ##############
    display_prediction(model, X_test)

    return 0


if __name__ == '__main__':
    main()