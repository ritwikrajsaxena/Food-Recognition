**Food Image Classification with ResNet50**

This project implements a food image classification system using a ResNet50 backbone with a custom classifier head inspired by CLIP's architecture.

Key Features
Modified ResNet50 Architecture: Uses pretrained ResNet50 weights with a custom classifier head (2048 → 512 → num_classes)

Advanced Data Augmentation: Includes random crops, flips, rotations, and color jittering

Comprehensive Training Metrics:

Accuracy, F1, Precision, Recall

AUROC (Area Under ROC Curve)

Training Optimizations:

Mixed precision training

Cosine annealing learning rate scheduling

Early stopping

Visualization: Detailed training curves for all metrics

Requirements:

Python 3.7+

PyTorch 1.12+

torchvision

albumentations

scikit-learn

tqdm

numpy

matplotlib

Installation

bash

pip install torch torchvision albumentations scikit-learn tqdm numpy matplotlib


Dataset Structure

Organize your dataset in this structure:

text

dataset/

    class1/
    
        image1.jpg
        
        image2.jpg
        
        ...
    
    class2/
        
        image1.jpg
        
        image2.jpg
        
        ...
    
    ...


Usage

For Google Colab:

Mount your Google Drive containing the dataset

Run all cells (the script handles Colab-specific setup)


For Local Execution:

Remove the Google Drive mounting code

Update dataset paths to local paths


Run:

bash

python food_classifier.py

Configuration-
Modify these parameters in the config dictionary:

python

config = {
    
    'patience': 3,          # Early stopping patience
    
    'max_epochs': 100,      # Maximum training epochs
    
    'batch_size': 32,       # Batch size
    
    'learning_rate': 0.00015, # Learning rate
    
    'num_workers': 4,       # Data loader workers
    
    'model_save_path': 'best_model.pth'  # Model save path

}


Key Components-

Data Augmentation:

Training: Random crops, flips, rotations, color jitter

Testing: Center crop only


Model Architecture:

ResNet50 backbone (pretrained on ImageNet)

Custom classifier head with dropout


Training Features:

Automatic mixed precision

Cosine learning rate annealing

Comprehensive metrics tracking

Model checkpointing


Expected Output-

The script will display:

Per-epoch training statistics

Validation metrics

Early stopping notifications

Final test set performance

Training curves visualization

Performance Optimization Tips


For faster training:

Increase batch_size if GPU memory allows

Use more num_workers for data loading


For better accuracy:

Try different learning rates

Adjust augmentation intensity

Increase model capacity


Customization-

To modify the model architecture:

python

class ResNet50CLIPStyle(nn.Module):

    def __init__(self, num_classes):
    
        super().__init__()
        
        self.backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        
        self.feature_extractor = nn.Sequential(*list(self.backbone.children())[:-1])
        
        self.classifier = nn.Sequential(
        
            nn.Linear(2048, 512),  # Modify these dimensions
            
            nn.ReLU(),
            
            nn.Dropout(0.2),       # Adjust dropout rate
            
            nn.Linear(512, num_classes)
        
        )


License: This project is provided under the MIT License.


**Food Image Classification with CLIP**

This project implements a food image classification system using OpenAI's CLIP 

(Contrastive Language-Image Pretraining) model as a feature extractor, with a custom classifier head for fine-grained food classification.


Features:

Utilizes CLIP's vision transformer (ViT-B/32) for image feature extraction

Custom classifier head with dropout for regularization

Advanced data augmentation pipeline using Albumentations


Comprehensive training metrics tracking:

Training/validation accuracy

F1 score, precision, recall

AUROC (Area Under ROC Curve)

Early stopping with model checkpointing

Mixed precision training for improved performance


Detailed training progress visualization-

Requirements:

Python 3.7+

PyTorch 1.7+

OpenAI CLIP

Albumentations

scikit-learn

tqdm

numpy

matplotlib


Installation:

bash

pip install torch torchvision clip albumentations scikit-learn tqdm numpy matplotlib


Usage-

Prepare your dataset in the following structure:

text

dataset/
    
    class1/
        
        image1.jpg
        
        image2.jpg
        
        ...
    
    class2/
        
        image1.jpg
        
        image2.jpg
        
        ...
    
    ...


Update the configuration in the script as needed:

python

config = {

    'patience': 3,
    
    'max_epochs': 100,
    
    'batch_size': 32,
    
    'learning_rate': 0.00001,
    
    'num_workers': 4,
    
    'model_save_path': 'best_model.pth'

}


Run the training script:

python

python train.py


Training Process-

The training process includes:

Automatic mixed precision training

Cosine annealing learning rate scheduling

Early stopping based on validation accuracy

Comprehensive metrics logging

Model checkpointing


Results-

The script outputs:

Per-epoch training statistics

Final test metrics including:

Accuracy

F1 score

Precision

Recall

AUROC

Training time statistics

Visualization of training metrics


Customization-

You can easily customize:

Data augmentation pipeline in get_train_transforms() and get_test_transforms()

Model architecture by modifying the classifier head

Training parameters in the config dictionary


Notes:

The script includes a workaround for a known CUDA/OpenMP compatibility issue

For best performance, run on a GPU with CUDA support

The model uses CLIP's pretrained weights and fine-tunes only the classifier head


License: This project is provided as-is under the MIT License.




**Both the files are designed to run on Google Colab.**

Google Drive Integration:

python

from google.colab import drive

drive.mount('/content/drive')

This explicitly mounts Google Drive, which is a Colab-specific feature.


CUDA/GPU Focus:

The code prioritizes GPU usage (device = "cuda" if torch.cuda.is_available() else "cpu") and includes a workaround for a CUDA/OpenMP issue common in Colab environments:

python

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Colab-specific workaround


Assumed File Paths:

The dataset path (/content/drive/MyDrive/dataset) follows Colab’s default Google Drive mount location.


Platform Compatibility-

While optimized for Colab, the code can also run on:

Local Python (with modifications): Remove the Google Drive mounting code, Update dataset paths to local paths (e.g., ./dataset), Ensure CUDA is available if using a GPU locally.

Other Cloud Platforms (e.g., AWS SageMaker, Kaggle Kernels): Replace Google Drive paths with platform-specific storage (e.g., S3 buckets for AWS), Adjust environment variables as needed.


Key Dependencies-

The script requires:

Python 3.7+

PyTorch (with CUDA if using GPU)

CLIP (pip install git+https://github.com/openai/CLIP.git)

Albumentations (for augmentations)

scikit-learn (for metrics)


How to Run Elsewhere-

Local Machine:

bash

pip install -r requirements.txt  # Create this file from the dependencies

python script.py  # Modify paths first


Kaggle:

Use the Food101 dataset directly from Kaggle or Upload the dataset to Kaggle Datasets.

Replace Google Drive paths with Kaggle paths (e.g., /kaggle/input/dataset).


AWS/GCP:

Use cloud storage (S3, GCS) and update paths accordingly.

Ensure GPU instances are selected for training.


Colab-Specific Advantages:

Free GPU access (T4/K80).

Pre-installed libraries (PyTorch, CUDA).

Seamless Google Drive integration.


