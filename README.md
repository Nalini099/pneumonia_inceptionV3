### Pneumonia Detection Using Inception-V3 on PneumoniaMNIST
## OBJECTIVE
To fine-tune a pre-trained Inception-V3 model using the PneumoniaMNIST dataset to classify chest X-rays into "pneumonia" or "normal" categories. The project involves implementing transfer learning, mitigating class imbalance, and preventing overfitting while ensuring high generalization performance.
## APPROACH
- **Transfer Learning**: Utilized Inception-V3 pre-trained on ImageNet. Final classification layer replaced with a two-class output layer.
- **Class Imbalance Handling**: Applied both weighted loss and WeightedRandomSampler for oversampling the minority class.
- **Overfitting Prevention**: Used data augmentation (rotation, flip, zoom), dropout, and weight decay.

## Evaluation Metrics
1. **Accuracy** – Overall model correctness.
2. **Recall (Sensitivity)** – Ensures pneumonia cases are not missed.
3. **ROC-AUC Score** – Measures separability between classes and handles imbalance effectively.

# Setup Instructions
In Google Colab:
Upload the PneumoniaMNIST zip file using the file upload dialog (click the folder icon on the left panel → upload).
Use the following code to extract it:

`
from google.colab import files
import zipfile
uploaded = files.upload()
for fname in uploaded:
    with zipfile.ZipFile(fname, 'r') as zip_ref:
        zip_ref.extractall('data') `

* Ensure that the folder structure after extraction is:
Additionally, for faster training performance in Google Colab, it is recommended to enable GPU acceleration:
Navigate to: `Runtime > Change runtime type > select GPU (preferably T4, if available)`

data/train/NORMAL  
data/train/PNEUMONIA  
data/test/NORMAL  
data/test/PNEUMONIA  
1. **INSTALL DEPENDENCIES**
`pip install -r requirements.txt`
2. **DATA SETUP**
Download the PneumoniaMNIST dataset from Kaggle:
[https://www.kaggle.com/datasets/rijulshr/pneumoniamnist/data](url)
Place the extracted folders in:
`data/train/NORMAL
data/train/PNEUMONIA
data/test/NORMAL
data/test/PNEUMONIA`
# Training the Model
`python train.py --epochs 10 --batch_size 32 --lr 1e-4`
# Evaluating the Model
`python evaluate.py --model_path model_weights.pth`
# Saved Model
Trained weights are saved in `model_weights.pth`for deployment or inference.

# Hyperparameters

Learning Rate: 1e-4 – Fine-tuning with minimal disruption to pretrained features.
- **Batch Size** : 32 – Balanced memory efficiency and convergence.
- **Epochs** : 10 – Sufficient for convergence with early stopping.
- **Dropout** : 0.5 – Applied before final layer.
- **Weight Decay** : 1e-5 – Regularization to reduce overfitting.

## Results Summary
**Accuracy**: ~96–98%
**Recall**: ~95–97%
**AUC**: ~96–99%
Results may vary slightly depending on training configuration.

# Repository Structure
- data/                    # Place dataset here (not committed)
- train.py                 # Training script
- evaluate.py              # Evaluation script
- model.py                 # Inception-V3 model modifications
- requirements.txt         # Project dependencies
- README.md                # Project documentation
- hyperparameters.md       # Justification for hyperparameter choices (optional)








