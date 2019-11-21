# Data401Project4

To Do list:
 - 3 Caltech101 models
    - Each with a different preprocesssing (1. raw images 2. Edge detection 3. Wavelet Transform (see AlexHunterNet.ipynb for 2 and 3)
  - 6 AlexHunter models
    - 3 with the same preprocessing above AND transfer learning, 3 with the preprocessing above AND NO transfer learning
    
Caltech101 Weight Creation.py already builds a model for Caltech101 and just needs to be adapted to include all preprocessing types
 - The deliverable for this script is a set of weights denoted 'caltech101_[preprocessingtype].h5' and reports on its performace

AlexHunterNet.ipynb needs to build a model on top of Caltech101 as well as just an independent model
 - The deliverable for this script/notebook is Performance (Precision, Recall, F1, Accuracy) for each model
