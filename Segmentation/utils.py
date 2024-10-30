import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

SMOOTH = 1.


def preprocess_image(img, mask):
    # resize image and mask
    img = tf.image.resize(img, (128, 128), method="nearest")
    mask = tf.image.resize(mask, (128, 128), method="nearest")

    # normalize image and mask
    img = tf.cast(img, tf.float32) / 255.0

    threshold = 180
    mask_thres = np.zeros(mask.shape).astype('uint8')
    mask_thres[mask < threshold] = 0
    mask_thres[mask > threshold] = 255
    mask = mask_thres
    mask = tf.cast(mask, tf.float32) / 255.0

    return img, mask


def plot_image_mask(img, mask):
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap='gray')
    plt.show()


def data_generator(image_dir_path, masks_dir_path, batch_size, seed, rotation_check=True):
    if rotation_check:
        data_gen_args = dict(
            horizontal_flip=True,
            vertical_flip=True,
            rotation_range=20
        )
    else:
        data_gen_args = dict(
            horizontal_flip=True,
            vertical_flip=True
        )
    imagegen = tf.keras.preprocessing.image.ImageDataGenerator(**data_gen_args)
    maskgen = tf.keras.preprocessing.image.ImageDataGenerator(**data_gen_args)

    train_image_gen = imagegen.flow_from_directory(
        image_dir_path,
        target_size=(128, 128),
        class_mode=None,
        color_mode='grayscale',
        batch_size=batch_size,
        shuffle=True,
        seed=seed
    )
    train_mask_gen = maskgen.flow_from_directory(
        masks_dir_path,
        target_size=(128, 128),
        class_mode=None,
        color_mode='grayscale',
        batch_size=batch_size,
        shuffle=True,
        seed=seed
    )

    train_gen = zip(train_image_gen, train_mask_gen)

    for (img, mask) in train_gen:
        img, mask = preprocess_image(img, mask)
        yield img, mask


def dice_coef(y_true, y_pred):
    y_true_f = tf.keras.layers.Flatten()(y_true)
    y_pred_f = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.math.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + SMOOTH) / (tf.math.reduce_sum(y_true_f) + tf.math.reduce_sum(y_pred_f) + SMOOTH)


def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


def plot_curves(loss, val_loss, dice, val_dice, epochs):
    epochs_range = range(epochs)

    fig = plt.figure(figsize=(12, 6))

    ax1 = plt.subplot(1, 2, 1)
    # ax1.set_ylim([0.93, 0.99])
    plt.plot(epochs_range, dice, label="train dice coef.")
    plt.plot(epochs_range, val_dice, label="validataion dice coef.")
    plt.title("Dice Coef.")
    plt.xlabel("Epoch")
    plt.ylabel("Dice Coef.")
    plt.legend(loc="lower right")

    ax2 = plt.subplot(1, 2, 2)
    # ax2.set_ylim([0, 0.5])
    plt.plot(epochs_range, loss, label="train loss")
    plt.plot(epochs_range, val_loss, label="validataion loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")

    fig.tight_layout()
    plt.show()
    # plt.savefig("unet_30epochs_loss_accuracy_drishtiGS1_OD_augmentation_2")
