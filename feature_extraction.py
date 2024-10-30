import os
import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt

from utils import preprocess_image_masks


def calculate_CDR(OC_mask, OD_mask):
    a = np.sum(OC_mask)
    b = np.sum(OD_mask)
    print(f"{a} / {b}")
    return a / b


def calculate_NRR(OD_mask, OC_mask):
    NRR = np.logical_xor(OD_mask, OC_mask)
    return NRR


def calculate_rim_area(NRR, mask):
    region = np.logical_and(NRR, mask)
    area = np.sum(region)
    return area


def calculate_rim_area_ratio(Ri, Rs, Rn, Rt):
    num = 1 + (Ri + Rs)
    den = 1 + (Rn + Rt)
    return float(num) / float(den)


def generate_ISNT_masks(img_height, img_width):
    S_region = np.zeros((img_height, img_width), dtype=np.uint8)

    center_x = img_width // 2
    center_y = img_height // 2

    # Create a meshgrid of coordinates
    y, x = np.ogrid[:img_height, :img_width]
    # Calculate the angle of each pixel relative to the center
    angle = np.arctan2(y - center_y, x - center_x) * 180 / np.pi
    # Normalize the angle to be between 0 and 360 degrees
    angle = (angle + 360) % 360

    # Set the pixels between 225 and 315 degrees to 1
    S_region[(angle >= 225) & (angle <= 315)] = 1
    # Other regions
    T_region = imutils.rotate(S_region, angle=90)
    I_region = imutils.rotate(T_region, angle=90)
    N_region = imutils.rotate(I_region, angle=90)

    return I_region, S_region, N_region, T_region


def calculate_feature_set(test_image_path, test_OD_mask_path, test_OC_mask_path,
                          images_list, OD_list, OC_list,
                          OD_model, OC_model,
                          I_region, S_region, N_region, T_region):
    # calculate CDR, NRR
    THRESHOLD = 0.5
    X = []
    num_images = len(images_list)

    for i in range(num_images):
        image_path = os.path.join(test_image_path, images_list[i])
        print(image_path)
        OD_path = os.path.join(test_OD_mask_path, OD_list[i])
        OC_path = os.path.join(test_OC_mask_path, OC_list[i])

        # read the images and masks and expand their dimensions so that we can preprocess them
        img = cv2.imread(image_path, 0)
        OD = cv2.imread(OD_path, 0)
        OC = cv2.imread(OC_path, 0)

        img = np.expand_dims(img, axis=2)
        OD = np.expand_dims(OD, axis=2)
        OC = np.expand_dims(OC, axis=2)

        # preprocess the image and masks for prediction
        img, OD, OC = preprocess_image_masks(img, OD, OC)

        # predict the masks using the previously saved model
        img_input = np.expand_dims(img, 0)

        OD_pred = OD_model.predict(img_input)
        OC_pred = OC_model.predict(img_input)

        OD_pred_threshold = OD_pred > THRESHOLD
        OC_pred_threshold = OC_pred > THRESHOLD

        OD_pred_threshold = OD_pred_threshold[0, :, :, 0]
        OC_pred_threshold = OC_pred_threshold[0, :, :, 0]

        # plot images
        plt.figure(figsize=(15, 15))

        plt.subplot(1, 6, 1)
        plt.title("Image")
        plt.imshow(img, cmap='gray')
        plt.axis("off")

        plt.subplot(1, 6, 2)
        plt.title("Ground Truth OC")
        plt.imshow(OC, cmap='gray')
        plt.axis("off")

        plt.subplot(1, 6, 3)
        plt.title("Predicted OC")
        plt.imshow(OC_pred_threshold, cmap='gray')
        plt.axis("off")

        plt.subplot(1, 6, 4)
        plt.title("Ground Truth OD")
        plt.imshow(OD, cmap='gray')
        plt.axis("off")

        plt.subplot(1, 6, 5)
        plt.title("Predicted OD")
        plt.imshow(OD_pred_threshold, cmap='gray')
        plt.axis("off")

        # calculate CDR (cup-disc ratio)
        CDR = calculate_CDR(OC_pred_threshold, OD_pred_threshold)
        print(f"CDR = {CDR}")

        # calculate NRR (Neuro-retinal Rim Ratio)
        NRR = calculate_NRR(OD_pred_threshold, OC_pred_threshold)

        # calculate ISNT areas
        Ri = calculate_rim_area(NRR, I_region)
        Rs = calculate_rim_area(NRR, S_region)
        Rn = calculate_rim_area(NRR, N_region)
        Rt = calculate_rim_area(NRR, T_region)

        print(f"Rim area of inferior region = {Ri}")
        print(f"Rim area of superior region = {Rs}")
        print(f"Rim area of nasal region = {Rn}")
        print(f"Rim area of temporal region = {Rt}")

        rim_area_ratio = calculate_rim_area_ratio(Ri, Rs, Rn, Rt)
        print(f"Rim area ratio = {rim_area_ratio}\n")

        plt.subplot(1, 6, 6)
        plt.title("NRR")
        plt.imshow(NRR, cmap='gray')
        plt.axis("off")

        plt.show()
        X.append([CDR, rim_area_ratio, Ri, Rs, Rn, Rt])

    return X
