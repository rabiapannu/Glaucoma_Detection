import os
import tensorflow as tf
from tensorflow.keras import optimizers
import argparse

from research_model import ResUNet
from utils import data_generator, dice_coef_loss, dice_coef, plot_image_mask, plot_curves

# TRAIN_IMAGE_PATH = "Drishti-GS/Train/Fundus_Images"
# TRAIN_MASK_PATH = "Drishti-GS/Train/Ground_Truths_OC"
# VAL_IMAGE_PATH = "Drishti-GS/Validation/Fundus_Images"
# VAL_MASK_PATH = "Drishti-GS/Validation/Ground_Truths_OC"

# MODEL_SAVE_PATH = "saved_models/ResUnet_Model_OC.keras"

SEED = 42
IMG_HEIGHT = 128
IMG_WIDTH = 128
IMG_CHANNELS = 1
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
N_CLASSES = 1
# BATCH_SIZE = 16
# EPOCHS = 50
# LR = 0.001


def plot_sample_images(img_gen, n=3):
    x, y = next(img_gen)
    for i in range(0, n):
        image = x[i, :, :, 0]
        mask = y[i, :, :, 0]
        plot_image_mask(image, mask)


def train_model(train_img_path, train_mask_path, val_img_path, val_mask_path, model_save_path,
                batch_size, epochs, lr):
    # image generators
    train_img_gen = data_generator(train_img_path, train_mask_path, BATCH_SIZE, SEED)
    val_img_gen = data_generator(val_img_path, val_mask_path, BATCH_SIZE, SEED)
    plot_sample_images(train_img_gen)

    # define and load model metrics
    num_train_imgs = len(os.listdir(os.path.join(train_img_path, "Train")))
    num_val_imgs = len(os.listdir(os.path.join(val_img_path, "Validation")))
    STEPS_PER_EPOCH = num_train_imgs // batch_size
    VAL_STEPS_PER_EPOCH = num_val_imgs // batch_size

    # Build and compile model
    resunet = ResUNet(INPUT_SHAPE)
    model = resunet.model
    adam = optimizers.Adam(lr)
    model.compile(optimizer=adam, loss=dice_coef_loss, metrics=[dice_coef])
    model.summary()
    tf.keras.utils.plot_model(model, show_shapes=True)

    # Train Model
    history = model.fit(
        train_img_gen,
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=epochs,
        verbose=1,
        validation_data=val_img_gen,
        validation_steps=VAL_STEPS_PER_EPOCH
    )
    model.save(model_save_path)

    # plot the training and validation dice_coefficient and loss at each epoch
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    dice = history.history['dice_coef']
    val_dice = history.history['val_dice_coef']
    plot_curves(loss, val_loss, dice, val_dice, EPOCHS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch-size', type=int, help="Batch Size")
    parser.add_argument('-ep', '--epochs', type=int, help="No. of Epochs")
    parser.add_argument('-lr', '--learning-rate', type=float, help="Learning Rate")
    parser.add_argument('-train-img-path', type=str, help="Path to Train Images Directory", required=True)
    parser.add_argument('-train-mask-path', type=str, help="Path to Train Masks Directory", required=True)
    parser.add_argument('-val-img-path', type=str, help="Path to Validation Images Directory", required=True)
    parser.add_argument('-val-mask-path', type=str, help="Path to Validation Masks Directory", required=True)
    parser.add_argument('-model-name', type=str, help="Path/Name.keras (filename to save your trained model)",
                        required=True)

    arg_list = parser.parse_args()
    BATCH_SIZE = arg_list.batch_size if arg_list.batch_size is not None else 16
    EPOCHS = arg_list.epochs if arg_list.epochs is not None else 50
    LR = arg_list.learning_rate if arg_list.learning_rate is not None else 0.001
    TRAIN_IMAGE_PATH = arg_list.train_img_path
    TRAIN_MASK_PATH = arg_list.train_mask_path
    VAL_IMAGE_PATH = arg_list.val_img_path
    VAL_MASK_PATH = arg_list.val_mask_path
    MODEL_SAVE_PATH = arg_list.model_name

    train_model(TRAIN_IMAGE_PATH, TRAIN_MASK_PATH, VAL_IMAGE_PATH, VAL_MASK_PATH, MODEL_SAVE_PATH,
                BATCH_SIZE, EPOCHS, LR)
