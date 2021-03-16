
## This Model is a modification for the yolov5 trained for a custom dataset of 3D Objects. 

This is application of YOLO in Robotics (Peg in Hole task)using KUKA iiwa robot for detection of the 3D coordinates of the object with respect to a camera frame (Logitech camera) using it's Interinsic parameters from calibration data and Also using Hough algorithim to detect the exact hole position.

Experiments and all models details [Weights & Biases](https://wandb.ai/) : you can found more about specific model reports [Runs of yolo and full results details](https://wandb.ai/mostafa_metwally/YOLOv5?workspace=user-mostafa_metwally)


You can also find the training dataset at the roboflow website and download the [Dataset](https://app.roboflow.com/ds/SvVogYrMHF?key=w17XxVDTQB
) or you could use this link in jupyter or other notebooks
- this dataset contains different images with lots of augmentations.
```bash
$ !curl -L "https://app.roboflow.com/ds/PHSnxXecjz?key=aY0NGwMCE3" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip
```

## Check for all the Requirements below or forked repo

Python 3.8 or later with all [requirements.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt) dependencies installed, including `torch>=1.7`. To install run:
```bash
$ pip install -r requirements.txt
```


# this yolov5 fork has different [detect.py](https://github.com/mostafa-metwaly/yolov5/blob/439c503c209d73fcb574950528d6e24c0b9d2c53/detect.py) file (where we added part for modified camera parameters and another part for exporting a txt file with the values of object x,y,z wrt the camera)
- you will need the detect file and the interinsic parameters of the camera.npy
- you will need the weights for the latest trained model(exp60) in the runs/train/exp60/weights/best.pt   (this is the model trained on yolov5x    with best results up till now)
- if you want to train a new model you will need the data itself with labels and it could be found in the data folder under (train and valid) you will also need the modified yaml file (data3.yaml) and maybe tune the hyperparameters (hyp.scratch.yaml)



## Trainning the model with hyperparameters.
```bash
$ python train.py --img 640 --batch 16 --epochs 150 --data data3.yaml --weights yolov5x.pt --hyp data/hyp.scratch.yaml
``` 

## Testing the model performance on specific dataset
```bash
$ python test.py --weights yolov5s.pt --data data3.yaml --source ./data/images/
```


## Detection of the objects:
run this on the given path of static folder containing images to be detected  ---> the results will be saved in the runs detect folder in a new exp.
```bash
$ python detect.py --weights yolov5s.pt --img 640 --source ./data/images/
```

using the weights from the specific training experiment (you could use best{2,3,60} from home these were old models), minimum confidence level 0.6 , source(external camera 2 or1), show the image of the detected objects , show only class 1(cylinder), choose the processing resource either cpu or default gpu(0,1,2)




```bash
$ python detect.py --weights ./runs/train/exp60/weights/best.pt --conf 0.6 --source 2 --view-img --classes 1

$ python detect.py --weights best60.pt --conf 0.8 --source 2 --view-img --classes 1 --save-txt --save-conf --device cpu
```


