# Glaucoma_Detection
Segmentation of Optic Disc (OD) and Optic Cup (OC), feature extraction, and glaucoma classification using CDR and NRR metrics. <br/>

## Data preparation 
Drishti-GS dataset was used for this research.<br/>
The folder structure should be as follows: <br/><br/>

<pre>
│Drishti-GS/<br/>
├──Drishti-GS/Drishti-GS1_diagnosis.xlsx<br/>
├──Train/<br/>
│  ├── Fundus_Images/<br/>
│  ├───├── Train/<br/>
│  │   ├────├── 002.png<br/>
│  │   ├────├── 004.png<br/>
│  │   ├────├── ...<br/>
│  ├── Ground_Truths\_OC/<br/>
│  ├───├── Train/<br/>
│  │   ├────├── 002.png<br/>
│  │   ├────├── 004.png<br/>
│  │   ├────├── ...<br/>
│  ├── Ground_Truths\_OD/<br/>
│  ├───├── Train/<br/>
│  │   ├────├── 002.png<br/>
│  │   ├────├── 004.png<br/>
│  │   ├────├── ...<br/>
│  ├── ......<br/>
├──Validation/<br/>
│  ├── Fundus_Images/<br/>
│  ├───├── Validation/<br/>
│  │   ├────├── 001.png<br/>
│  │   ├────├── 006.png<br/>
│  │   ├────├── ...<br/>
│  ├── Ground_Truths\_OC/<br/>
│  ├───├── Validation/<br/>
│  │   ├────├── 001.png<br/>
│  │   ├────├── 006.png<br/>
│  │   ├────├── ...<br/>
│  ├── Ground_Truths\_OD/<br/>
│  ├───├── Validation/<br/>
│  │   ├────├── 002.png<br/>
│  │   ├────├── 006.png<br/>
│  │   ├────├── ...<br/>
│  ├── ......<br/>
├──Test/<br/>
│  ├── Fundus_Images/<br/>
│  ├───├── Test/<br/>
│  │   ├────├── 003.png<br/>
│  │   ├────├── 005.png<br/>
│  │   ├────├── ...<br/>
│  ├── Ground_Truths\_OC/<br/>
│  ├───├── Test/<br/>
│  │   ├────├── 003.png<br/>
│  │   ├────├── 005.png<br/>
│  │   ├────├── ...<br/>
│  ├── Ground_Truths\_OD/<br/>
│  ├───├── Test/<br/>
│  │   ├────├── 003.png<br/>
│  │   ├────├── 005.png<br/>
│  │   ├────├── ...<br/>
│  ├── ......<br/>
</pre>
<br/>

## Segmentation

The _segmentation/_ folder has two scripts named as **train.py** and **test.py**. <br/>

These two scipts can be run to train and test Optic Disc (OD) and Optic Cup (OC) segmentation respectively. <br/>
For example to train the model for OC segmentation, run the following script on terminal: <br/>

python Segmentation/train.py -train-img-path "Drishti-GS/Train/Fundus_Images/" -train-mask-path "Drishti-GS/Train/Ground\_Truths\_OC/" -val-img-path "Drishti-GS/Validation/Fundus\_Images/" -val-mask-path "Drishti-GS/Validation/Ground\_Truths\_OC/" -saved-model "saved\_models/OC_model.keras"

<br/>

## Feature Extraction and Classification

After successfully training and saving the models for OC and OD, run the **main.py** script to calculate the cup-to-disc ratio (CDR) and neuro-retinal rim ratio (NRR) to feed it into the classification model for the detection of glaucoma. <br/>

Run the main.py script as follows: <br/>

python main.py --OC-model "saved_models/OC\_model.keras" --OD-model "saved_models/OD\_model.keras" --dataset "Path/to/Dataset_directory" <br/>

