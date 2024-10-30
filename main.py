import os.path

import numpy as np
import pandas as pd
import argparse
from keras.models import load_model
from sklearn import svm
from sklearn import metrics

from utils import plot_ISNT_masks, plot_confusion_matrix, get_data
from feature_extraction import generate_ISNT_masks, calculate_feature_set

"""OC_MODEL_PATH = "saved_models/ResUnet_OC.keras"
OD_MODEL_PATH = "saved_models/ResUnet_OD.keras"

DATASET_EXCEL_FILE_LABELS = "Drishti-GS/Drishti-GS1_diagnosis.xlsx"

TRAIN_IMAGE_PATH = "Drishti-GS/Train/Fundus_Images/Train"
TRAIN_OD_MASK_PATH = "Drishti-GS/Train/Ground_Truths_OD/Train"
TRAIN_OC_MASK_PATH = "Drishti-GS/Train/Ground_Truths_OC/Train"

TEST_IMAGE_PATH = "Drishti-GS/Test/Fundus_Images/Test"
TEST_OD_MASK_PATH = "Drishti-GS/Test/Ground_Truths_OD/Test"
TEST_OC_MASK_PATH = "Drishti-GS/Test/Ground_Truths_OC/Test"
"""
IMG_HEIGHT = 128
IMG_WIDTH = 128


def get_X_and_Y(image_dir, OD_dir, OC_dir, OD_model, OC_model):
    # generate the ISNT masks
    I_region, S_region, N_region, T_region = generate_ISNT_masks(IMG_HEIGHT, IMG_WIDTH)
    plot_ISNT_masks(I_region, S_region, N_region, T_region)

    # calculate feature set X, and outputs Y
    images_list, OD_list, OC_list = get_data(image_dir, OD_dir, OC_dir)
    img_list_numbers = [image.split('.')[0] for image in images_list]
    df_subset = df[df["Drishti-GS File"].str.extract(r'(\d+)')[0].isin(img_list_numbers)]
    df_subset["Total"] = df_subset["Total"].map({"Glaucomatous": 1, "Normal": 0})

    X = calculate_feature_set(image_dir, OD_dir, OC_dir,
                              images_list, OD_list, OC_list,
                              OD_model, OC_model,
                              I_region, S_region, N_region, T_region)
    X = np.asarray(X)
    Y = df_subset["Total"]
    return X, Y


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--OC-model", type=str, help="Path to OC model.keras file", required=True)
    parser.add_argument("--OD-model", type=str, help="Path to OD model.keras file", required=True)
    parser.add_argument("--dataset", type=str, help="Path to dataset directory", required=True)

    arg_list = parser.parse_args()
    OC_MODEL_PATH = arg_list.OC_model
    OD_MODEL_PATH = arg_list.OD_model
    DATASET_DIR = arg_list.dataset

    TEST_IMAGE_PATH = os.path.join(DATASET_DIR, "Test/Fundus_Images/Test")
    TEST_OD_MASK_PATH = os.path.join(DATASET_DIR, "Test/Ground_Truths_OD/Test")
    TEST_OC_MASK_PATH = os.path.join(DATASET_DIR, "Test/Ground_Truths_OC/Test")
    TRAIN_IMAGE_PATH = os.path.join(DATASET_DIR, "Train/Fundus_Images/Train")
    TRAIN_OD_MASK_PATH = os.path.join(DATASET_DIR, "Train/Ground_Truths_OD/Train")
    TRAIN_OC_MASK_PATH = os.path.join(DATASET_DIR, "Train/Ground_Truths_OC/Train")
    DATASET_EXCEL_FILE_LABELS = os.path.join(DATASET_DIR, "Drishti-GS1_diagnosis.xlsx")

    # load the models for prediction of OC and OD masks
    OC_model = load_model(OC_MODEL_PATH, compile=False)
    OD_model = load_model(OD_MODEL_PATH, compile=False)

    # load labels into pandas dataframe
    df = pd.read_excel(DATASET_EXCEL_FILE_LABELS)
    df = df.dropna(how="all").dropna(how="all", axis=1)
    headers = df.iloc[0]
    df = pd.DataFrame(df.values[1:], columns=headers)
    print(df.head())

    # get test and train data set
    X_test, Y_test = get_X_and_Y(TEST_IMAGE_PATH, TEST_OD_MASK_PATH, TEST_OC_MASK_PATH,
                                 OD_model, OC_model)
    X_train, Y_train = get_X_and_Y(TRAIN_IMAGE_PATH, TRAIN_OD_MASK_PATH, TRAIN_OC_MASK_PATH,
                                   OD_model, OC_model)

    # SVM Training
    model_SVM = svm.SVC(kernel="rbf")
    history = model_SVM.fit(X_train, Y_train)

    # predict using SVM model
    Y_pred = model_SVM.predict(X_test)
    accuracy = metrics.accuracy_score(Y_test, y_pred=Y_pred)
    print(f"Accuracy = {accuracy}")
    precision = metrics.precision_score(Y_test, y_pred=Y_pred)
    print(f"Precision = {precision}")
    recall = metrics.recall_score(Y_test, y_pred=Y_pred)
    print(f"Recall = {recall}")

    # Confusion matrix
    plot_confusion_matrix(Y_test, Y_pred)
