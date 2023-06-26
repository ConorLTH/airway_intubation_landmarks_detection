# airway_intubation_landmarks_detection
  This is the code release for the Airway Intubation Landmarks Detection.

## Train 
To run the training of the model created in the work, please follow the below process:

Firstly, the anaconda is required and environment can be created:
```
conda create -n samdef python=3.9
conda activate samdef
```

And then to install pytorch:
```
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
```
Please ensure your device can support cuda. The cpu version of MMdetection may introduce errors.

  To install MMdetection:
```
pip install mim
pip install mmengine
pip install mmcv-full=1.7
```
  In our experiments, we run the previous version of mmcv-full to build models. The newer version of mmdetection has made a great change of mmcv which may also introduce errors.
  MMdetection run training and testing of models with config files.
  In this repository, the config files used in experiment of this work are also provided.

### To train the model:
Single GPU: 
```
python tools/train.py [config_file]
```
Multiple GPU: 
```
bash tools/dist_train.sh [config_file]  [NUM_GPU]
```
  The [config_file] means the path to the config file.
  The [NUM_GPU] means the number of GPU to be used for training.
  To get more details, please refer to the official documents of MMdetection.

### To test the model:
To test model:
```
python tools/test.py [config_file] [checkpoint]
```
  The [config_file] the config file also used in training.
  The [checkpoint] is the pth file storing the training information. 
