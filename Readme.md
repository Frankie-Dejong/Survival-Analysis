# Cancer Prediction

Our work takes inspiration from the paper *Histopathology images predict multi-omics  aberrations and prognoses in colorectal  cancer patients*, and make some essential changes for better performance.

## Requirements


```shell
conda create --name your_env_name python=3.10
conda activate your_env_name
pip install -r requirements.txt
```

### Openslide

```
pip install Openslide-python
```

#### For Windows:

Download from [OpenSlide](https://openslide.org/download/), add **bin** and **lib** to environment variables.

If you encounter any errors during the `import openslide` process, find the file lowlevel.py from your error message, then add the code following:

```python
import os
os.add_dll_directory("<your openslide bin path>")
```

#### For Ubuntu:

```
sudo apt install openslide-tools
```

## Data Preparation

### TCGA

Data source: [GDC Data Portal](https://portal.gdc.cancer.gov/)

Filter: 
- Cases
    - Primary Site: colon, rectum
    - Program: TCGA
    - Project: TCGA-COAD, TCGA-READ
- Files
    - Data Type: Slide Image
    - Experimental Strategy: Tissue Slide

Download tool: [gdc-data-transfer-tool](https://gdc.cancer.gov/access-data/gdc-data-transfer-tool)

## Data Proprocessing
To run our code, change the configuration at configuration/tcga_cfg.yaml.
### Processing TCGA

To get norms from TCGA slides, run
```
python data_preprocessing/main.py --norm_only
```

To pretrain our cnn model for feature extraction, run

```
python pretrain/train.py
```
## Train
To train our model for survival prediction, run
```
python survival_prediction/train.py
```

## Evaluation

To evaluate our model, run
```
python survival_prediction/eval.py
```
