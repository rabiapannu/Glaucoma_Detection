# Glaucoma_Detection
Segmentation of Optic Disc (OD) and Optic Cup (OC), feature extraction, and glaucoma classification using CDR and NRR metrics. <br/>

## Data preparation 
Drishti-GS dataset was used for this research.<br/>
The folder structure should be as follows: <br/><br/>

<pre>
│Drishti-GS/
├──Drishti-GS/Drishti-GS1_diagnosis.xlsx
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

The _segmentation/_ folder has two scripts named as **train.py** and **test.py**. <br/>

These two scipts can be run to train and test Optic Disc (OD) and Optic Cup (OC) segmentation respectively. <br/>
For example to train the model for OC segmentation, run the following script on terminal: <br/>

<pre>
python Segmentation/train.py -train-img-path "Drishti-GS/Train/Fundus_Images/" -train-mask-path "Drishti-GS/Train/Ground_Truths_OC/" -val-img-path "Drishti-GS/Validation/Fundus_Images/" -val-mask-path "Drishti-GS/Validation/Ground_Truths_OC/" -saved-model "saved_models/OC_model.keras"
</pre>

## Feature Extraction and Classification

After successfully training and saving the models for OC and OD, run the **main.py** script to calculate the cup-to-disc ratio (CDR) and neuro-retinal rim ratio (NRR) to feed it into the classification model for the detection of glaucoma. <br/>

Run the main.py script as follows: <br/>

<pre>
python main.py --OC-model "saved_models/OC_model.keras" --OD-model "saved_models/OD_model.keras" --dataset "Path/to/Dataset_directory" <br/>
</pre>
