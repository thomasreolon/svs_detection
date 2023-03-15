[![Maintenance](https://img.shields.io/badge/Maintained%3F-No-red.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity) [![Generic badge](https://img.shields.io/badge/python-3.5+-blue.svg)](https://shields.io/) [![Generic badge](https://img.shields.io/badge/version-v1.0-cc.svg)](https://shields.io/)

# Detection and Tracking Using Low Power Devices

<!-- ##### :speech_balloon: A report explaining the project can be found here: [report.pdf](report.pdf) :speech_balloon: -->

Codebase for my Master Thesis at unitn: optimizing a motion map simulation algorithm and a CNN to perform object detection with low energy comsumption.
This repo contains:
- the code to find the best configurations for the simulator
- the code to train the  CNN (architecture from YOLO & PhiNet) 
- the code to optimize the CNN's architecture

<!-- ### Example of 3D scene reconstruction
Starting from 8 views of the same scene, the model reconstructs the scene:

![image](https://media.giphy.com/media/WrxRcc5mnexksBeHyd/giphy.gif) -->

<!-- table with MaP -->

## Set Up

##### Install environment requirements

... [requirements.txt TODO]

##### Download the dataset

... [upload GDrive TODO]

##### Reproduce results

```sh
_scripts.run_experiments.bat  # windows
```


## Code Structure

```
main.py --> trains a CNN
engine.py --> function called from main.py
configs/defaults.py --> settings to run main.py
policy_learn.py --> finds the best configurations for the simulator & generates policies
_scripts/** --> other less important things
```


## Aknowledgements

Parts of the code are from [yolov5](https://github.com/ultralytics/yolov5) and [phinets](https://github.com/fpaissan/micromind).

