[![Maintenance](https://img.shields.io/badge/Maintained%3F-No-red.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity) [![Generic badge](https://img.shields.io/badge/python-3.5+-blue.svg)](https://shields.io/) [![Generic badge](https://img.shields.io/badge/version-v1.0-cc.svg)](https://shields.io/)

# Detection and Tracking Using Low Power Devices

##### :speech_balloon: A report explaining the project can be found here: [report.pdf](report.pdf) :speech_balloon:

Codebase for my Master Thesis at unitn: optimizing a motion map simulation algorithm and a CNN to perform object detection with low energy comsumption.
This repo contains:
- the code to find the best configurations for the simulator
- the code to train the  CNN (architecture from YOLO & PhiNet) 
- the code to optimize the CNN's architecture

### Example of Detection with motion maps and light neural network

![image](https://media.giphy.com/media/GdyxC4fwuIUanCbqqa/giphy.gif)

<!-- table with MaP -->

## Set Up


##### Download the dataset

dataset : https://drive.google.com/file/d/1QsHU9vBjj26ZLf_g7w6fozb1OTNEvO19

The dataset contains videos from: **MOT17**, **MOT**, **Streets23** (and poorly annotated highways webcams)

After having unzipped the files, update mot_path inside configs/defaults.py to

##### Install environment requirements

```sh
pip install -r requirements.txt
```

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

### Detection with motion maps
Starting from 8 views of the same scene, the model reconstructs the scene:

![image](https://media.giphy.com/media/32Nn2n0e26Hmuc3eOl/giphy.gif)
![image](https://media.giphy.com/media/h4AZ6RsxUMD0BuBQwo/giphy.gif)


## Aknowledgements

[yolov5](https://github.com/ultralytics/yolov5): base architecture

[phinets](https://github.com/fpaissan/micromind): detector backbone

