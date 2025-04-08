# Glaucoma_Detection
Segmentation of Optic Disc (OD) and Optic Cup (OC), feature extraction, and glaucoma classification using CDR and NRR metrics. <br/>

## Setup Instructions
To run this project, you will need:

- Python 3.7+
- TensorFlow 2.x
- OpenCV
- NumPy
- Scikit-learn
- Matplotlib

You can install the required dependencies using:

```bash
pip install -r requirements.txt

## Datasets Used
This research utilizes four publicly available retinal fundus image datasets:
- [Drishti-GS](https://cvit.iiit.ac.in/projects/mip/drishti-gs/mip-dataset2/Home.php)
- [RIM-ONE](https://github.com/miag-ull/rim-one-dl)
- [ORIGA](https://www.kaggle.com/datasets/arnavjain1/glaucoma-datasets)
- [REFUGE2](https://refuge.grand-challenge.org/Download/)
These datasets contain labeled optic disc and optic cup regions and have been widely used for glaucoma detection and segmentation tasks.<br/>

## Data preparation 
Ensure your dataset is organized in the following folder structure before running the code: <br/><br/>

<pre>
│RootFolder/
├──Train/
│  ├── Fundus_Images/
│  ├───├── Train/
│  │   ├────├── 002.png
│  │   ├────├── 004.png
│  │   ├────├── ...
│  ├── Ground_Truths\_OC/
│  ├───├── Train/
│  │   ├────├── 002.png
│  │   ├────├── 004.png
│  │   ├────├── ...
│  ├── Ground_Truths\_OD/
│  ├───├── Train/
│  │   ├────├── 002.png
│  │   ├────├── 004.png
│  │   ├────├── ...
│  ├── ......
├──Validation/
│  ├── Fundus_Images/
│  ├───├── Validation/
│  │   ├────├── 001.png
│  │   ├────├── 006.png
│  │   ├────├── ...
│  ├── Ground_Truths\_OC/
│  ├───├── Validation/
│  │   ├────├── 001.png
│  │   ├────├── 006.png
│  │   ├────├── ...
│  ├── Ground_Truths\_OD/
│  ├───├── Validation/
│  │   ├────├── 002.png
│  │   ├────├── 006.png
│  │   ├────├── ...
│  ├── ......
├──Test/
│  ├── Fundus_Images/
│  ├───├── Test/
│  │   ├────├── 003.png
│  │   ├────├── 005.png
│  │   ├────├── ...
│  ├── Ground_Truths\_OC/
│  ├───├── Test/
│  │   ├────├── 003.png
│  │   ├────├── 005.png
│  │   ├────├── ...
│  ├── Ground_Truths\_OD/
│  ├───├── Test/
│  │   ├────├── 003.png
│  │   ├────├── 005.png
│  │   ├────├── ...
│  ├── ......
</pre>

## Segmentation

The `Segmentation/` folder contains two key scripts: **`train.py`** and **`test.py`**, which are used to train and evaluate segmentation models for the **Optic Disc (OD)** and **Optic Cup (OC)** regions in fundus images. <br/>

### Segmentation Model
A modified **ResUNet** architecture is used for segmentation, leveraging residual connections along with the U-Net structure for better feature extraction.

The images and masks are resized to **128×128**, and the model is trained using the **Dice loss** to handle class imbalance in medical image segmentation.

### Parameters
You can adjust the following parameters:
- `--batch-size`: batch size for training (default: 8)
- `--epochs`: number of training epochs (e.g., 50)
- `--lr`: learning rate (e.g., 0.001)
- `--train-img-path`: path to training images
- `--train-mask-path`: path to corresponding masks
- `--val-img-path`: path to validation images
- `--val-mask-path`: path to validation masks
- `--saved-model`: path where the trained model will be saved

### Training Example (OC)
```bash
python Segmentation/train.py \
  --train-img-path "Drishti-GS/Train/Fundus_Images/" \
  --train-mask-path "Drishti-GS/Train/Ground_Truths_OC/" \
  --val-img-path "Drishti-GS/Validation/Fundus_Images/" \
  --val-mask-path "Drishti-GS/Validation/Ground_Truths_OC/" \
  --saved-model "saved_models/OC_model.keras" \
  --batch-size 8 \
  --epochs 50 \
  --lr 0.001

Similarly, you can also train for OD segmentation by changing the mask paths. <br/>

## Feature Extraction and Glaucoma Classification

After obtaining OD and OC segmentation masks using trained models, the `main.py` script performs the following steps:

1. **ISNT Region Masking**: Creates Inferior, Superior, Nasal, and Temporal (ISNT) region masks using the segmented OD and OC areas.
2. **Feature Extraction**: Computes key glaucoma indicators such as:
   - **CDR** (Cup-to-Disc Ratio)
   - **NRR** (Neuro-Retinal Rim Ratio)
3. **Classification**: Trains and evaluates a Support Vector Machine (SVM) with an RBF kernel to classify fundus images as **glaucomatous** or **non-glaucomatous** based on the extracted features.

### Run Classification

Use the following command to perform feature extraction and glaucoma classification:

```bash
python main.py \
  --OC-model "saved_models/OC_model.keras" \
  --OD-model "saved_models/OD_model.keras" \
  --dataset "Path/to/Dataset_directory"


## Reproducibility and Testing
To ensure consistent results across runs, we recommend the following:

- Use the same dataset structure as described above.
- Ensure consistent random seeds and hyperparameters in training scripts.
- Save trained models using clearly defined paths (e.g., `saved_models/OC_model.keras`).

Please keep in mind to run the segmentation scripts first and then run the main script feature extraction and classification.