CXR CLINICAL AI
MODULE 1.1 — VIEW CLASSIFICATION
LOCAL VS CODE DEPLOYMENT

============================================================
TASK
============================================================

This package contains the trained Module 1.1 chest X-ray
view classifier.

Input:
    Chest X-ray image

Output:
    AP
    Lateral

PA is NOT included.

============================================================
MODEL
============================================================

Architecture:

    DenseNet121
        |
        v
    Global Average Pooling
        |
        v
    Fully Connected Layer (2)
        |
        v
    AP / Lateral

The DenseNet121 backbone was initialized using ImageNet
pretrained weights during training.

============================================================
TRAINING
============================================================

Dataset:
    CXR 10K

Training images:
    7,000

Validation images:
    1,500

Test images:
    1,500

Classes:
    AP
    Lateral

PA:
    Excluded

Test accuracy:
    99.93 percent

Test F1:
    99.93 percent

============================================================
LOCAL SETUP
============================================================

Create a Python virtual environment in VS Code.

Then install:

    pip install -r requirements.txt

============================================================
RUN TEST
============================================================

From this package root:

    python test_inference.py

The script loads the trained DenseNet121 checkpoint and
runs predictions on the included sample images.

============================================================
MODEL OUTPUT
============================================================

Example:

    Predicted view: AP
    Confidence: 0.9998

============================================================
IMPORTANT
============================================================

The model checkpoint and configuration files must remain
together.

Do not rename:

    view_classifier_complete.pth
    model_config.json
    class_mapping.json

============================================================
NEXT PROJECT STEP
============================================================

After local validation of Module 1.1, the project will
continue with the disease prediction branch of Module 1.
