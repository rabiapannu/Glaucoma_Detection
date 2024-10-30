import matplotlib.pyplot as plt
import numpy as np
import argparse
from keras.models import load_model
from tensorflow.keras.metrics import MeanIoU

from utils import data_generator

# SAVED_MODEL_PATH = "saved_models/ResUnet_Model_OC.keras"
# TEST_IMAGE_PATH = "Drishti-GS/Test/Fundus_Images"
# TEST_MASK_PATH = "Drishti-GS/Test/Ground_Truths_OC"

# BATCH_SIZE = 16
SEED = 42


def plot_images_predictions(test_image_batch, test_mask_batch, y_pred, batch_size, threshold):
    for i in range(batch_size):
        test_img = test_image_batch[i]
        ground_truth = test_mask_batch[i]
        prediction = y_pred[i, :, :, :]
        predicted_img = (prediction[:, :, 0] > threshold).astype(np.uint8)

        print(i + 1)
        # j = i * 3

        plt.figure(figsize=(15, 15))

        plt.subplot(1, 3, 1)
        plt.title("Image")
        plt.imshow(test_img, cmap='gray')
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.title("Ground Truth")
        plt.imshow(ground_truth, cmap='gray')
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.title("Predicted Mask")
        plt.imshow(predicted_img, cmap='gray')
        plt.axis("off")

        plt.show()


def test_model(saved_model_path, test_image_path, test_mask_path, batch_size):
    model = load_model(saved_model_path, compile=False)

    # load data
    test_img_gen = data_generator(test_image_path, test_mask_path, batch_size, SEED, rotation_check=False)
    test_image_batch, test_mask_batch = test_img_gen.__next__()

    # predict
    THRESHOLD = 0.5
    y_pred = model.predict(test_image_batch)
    y_pred_threshold = y_pred > THRESHOLD
    print(y_pred_threshold.shape)

    # plot images
    plot_images_predictions(test_image_batch, test_mask_batch, y_pred, batch_size, THRESHOLD)

    # Calculate IOU (Intersection Over Union)
    n_classes = 2
    IOU_keras = MeanIoU(num_classes=n_classes)
    IOU_keras.update_state(y_pred_threshold, test_mask_batch)
    print("Mean IoU = ", IOU_keras.result().numpy())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch-size', type=int, help="Batch Size")
    parser.add_argument('-test-img-path', type=str, help="Path to Test Images Directory", required=True)
    parser.add_argument('-test-mask-path', type=str, help="Path to Test Masks Directory", required=True)
    parser.add_argument('-model-path', type=str, help="Path/Name.keras (Enter the path to your saved "
                                                      "model.keras file)", required=True)
    arg_list = parser.parse_args()
    BATCH_SIZE = arg_list.batch_size if arg_list.batch_size is not None else 16
    TEST_IMAGE_PATH = arg_list.test_img_path
    TEST_MASK_PATH = arg_list.test_mask_path
    SAVED_MODEL_PATH = arg_list.model_path

    test_model(SAVED_MODEL_PATH, TEST_IMAGE_PATH, TEST_MASK_PATH,
               BATCH_SIZE)
