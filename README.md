# Mushroom Multiclass Classification

## Requirement
Git clone:  `git clone https://github.com/IlianK/mushroom_classification.git`
 
Run: `pip install -r requirements.txt`

## Overview

### 1. `dataset`
This folder contains all data-related files, including:
- **Raw Data Images**: Original mushroom images provided by Kaggle.
- **Preprocessed Images**: Saved as tensors after preprocessing.
- **CSV Mappings**:
  - `train.csv`: Provided by Kaggle, containing 2371 labeled training images.
  - `test.csv`: Provided by Kaggle, containing unlabeled 600 test images.
  - `train_with_concept.csv`: Created for the Concept Bottleneck Model (CBM), mapping images to predefined concepts.

### 2. `exploration`
This folder contains outputs from data analysis and visualization, including:
- **Sample Images**: Representative images from each class.
- **Clustering Outputs**: Results from PCA, t-SNE, and UMAP clustering methods applied:
  - Per class clustering
  - Overall dataset clustering

### 3. `models`
This folder stores model-related artifacts, including:
- **Log Event Files**: Training logs and metrics.
- **Evaluation Results**: Performance metrics such as accuracy and loss.
- **Train/Validation Accuracy and Loss Plots**:
  - Baseline models
  - Fine-tuned models
  - Further enhanced AlexNet and ResNet models

### 4. `report`
This folder contains the final project documentation, including:
- **Paper Report**: The research paper.
- **Presentation Slides**: pptx. slides summarizing the project.


---

## Source Code (`src` folder)

### 1. `data_exploration.ipynb`
- Notebook for exploring the dataset.
- Utilizes `data_exploration.py` for:
  - Class distribution analysis
  - Image size distribution
  - Clustering techniques (PCA, t-SNE, UMAP)

### 2. `data_preparation.ipynb`
- Notebook for preparing data for training.
- Uses `data_preparation.py` to apply transformations, including:
  - Random horizontal flipping
  - Rotations up to 20 degrees

### 3. `training.ipynb`
- Notebook for finetuning baselines and enhanced models.
- Utilizes `training.py` for defining models with configurable parameters such as:
  - Batch size
  - Learning rate
  - Scheduler
  - Dropout rate (for custom models)

### 4. `training_concepts.ipynb`
- Implementation of the Concept Bottleneck Model (CBM).
- Provides functionality for:
  - Training the model with concept annotations
  - Performing intervention experiments

### 5. `explain_models.ipynb`
- Notebook for model interpretability analysis.
- Contains functions to load trained models and analyze them using:
  - Integrated Gradients for feature importance visualization.

### 6. `event_reader.ipynb`
- Notebook for exttracting events and displaying loss / accuracy plots of trained model logs.
- Utilizes `event_reader.py` for function definitions


## Model Setup and Training Configuration
The notebook `training.ipynb` is used to load and fine-tune various models using the following parameters:

```python
SET_GET_MODEL = 'GET'   # Options: 'SET' or 'GET'
WRITE = False           # True overwrites event files
FINETUNE = True         # True = fine-tuning

# Options: alexnet, resnet, vgg16, densenet, efficientnet
#          custom_alexnet, custom_resnet
model_type = 'custom_resnet'  
```

- **SET_GET_MODEL:**
  - When set to `'SET'`, the model is initialized with pre-trained weights (not fine-tuned) and prepared for training.
  - When set to `'GET'`, the fine-tuned model is loaded from the directory `BASELINE_FINE_DIR`.

- **Model Options:**
  - The baseline models can be used: `alexnet`, `resnet`, `vgg16`, `densenet`, `efficientnet`.
  - Two further optimized custom models: `custom_alexnet`, `custom_resnet`.



