# VAT-ABAW
Combine 3D-CNN and Transformer for Valence-Arousal estimation on the AffWild2 dataset.

## Getting Started

### Install requirements

```
pip install -r requirements.txt
```

### Prepare ABAW dataset

- First, apply for the newest version Aff-wild2 dataset.
URL: https://ibug.doc.ic.ac.uk/resources/fg-2020-competition-affective-behavior-analysis/

- Second, download the **cropped-aligned images** and **annotations**, and put them on the folder **data/**  of this project

- Third, run **data/generate_dataset.py**, then you should get a folder **data/dataset** which contains Training_Set and Validation_Set images and annotations, also the Test_Set images.

### Training
To train models, change paths in **mypath.py** with your ones, and run **train.py**
We have done several experiments on different models, I will continue to push them into this repo.

## Paper
Will be updated soon.
 
## Contact and information
please contact me:
hscheng@mail.bnu.edu.cn
