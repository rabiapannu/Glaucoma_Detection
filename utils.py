import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def get_data(images_path, od_masks_path, oc_masks_path):
    images_list = os.listdir(images_path)
    OD_list = os.listdir(od_masks_path)
    OC_list = os.listdir(oc_masks_path)

    images_list.sort()
    OD_list.sort()
    OC_list.sort()
    return images_list, OD_list, OC_list


def normalize_mask_helper(mask):
    threshold = 180
    mask_thres = np.zeros(mask.shape).astype('uint8')
    mask_thres[mask < threshold] = 0
    mask_thres[mask > threshold] = 255
    mask_normalized = mask_thres
    mask_normalized = tf.cast(mask_normalized, tf.float32) / 255.0
    return mask_normalized


def preprocess_image_masks(img, OD_mask, OC_mask):
    # resize image and mask
    img = tf.image.resize(img, (128, 128), method="nearest")
    OD_mask = tf.image.resize(OD_mask, (128, 128), method="nearest")
    OC_mask = tf.image.resize(OC_mask, (128, 128), method="nearest")

    # normalize image and mask
    img = tf.cast(img, tf.float32) / 255.0

    OD_mask = normalize_mask_helper(OD_mask)
    OC_mask = normalize_mask_helper(OC_mask)

    return img, OD_mask, OC_mask


def plot_ISNT_masks(I_region, S_region, N_region, T_region):
    plt.figure(figsize=(15, 15))

    plt.subplot(1, 4, 1)
    plt.imshow(S_region.squeeze(), cmap='gray')
    plt.title("Superior Region")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(T_region.squeeze(), cmap='gray')
    plt.title("Temporal Region")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(I_region.squeeze(), cmap='gray')
    plt.title("Inferior Region")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(N_region.squeeze(), cmap='gray')
    plt.title("Nasal Region")
    plt.axis("off")

    plt.show()


def plot_confusion_matrix(Y_test, Y_pred):
    cm = confusion_matrix(Y_test, Y_pred)
    print("Confusion Matrix:")
    print(cm)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model_SVM.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()
