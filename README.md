# airway_intubation_landmarks_detection
  This is the code release for the Airway Intubation Landmarks Detection.
  The repository is still under working

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
pip install mmdet
```
  In our experiments, we run the previous version of mmcv-full to build models. The newer version of mmdetection has made a great change of mmcv which may also introduce errors.
  MMdetection run training and testing of models with config files. Almost all of experiments of our work is run with MMdetection, and therefore, we recommend reading official documents of MMdetection before trying to run our code.
  In this repository, the config files used in experiment of this work are also provided. The file with name of *samdefdetr_config_nostril.py* is the config file for training and testing with the nostril dataset. And the other one with the name of *samdefdetr_config_glottis.py* is for glottis dataset. Please change the path inside the config file based on your own device. 
  
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

### To test the model:
To test model:
```
python tools/test.py [config_file] [checkpoint]
```
  The [config_file] the config file also used in training.
  The [checkpoint] is the pth file storing the training information. 
