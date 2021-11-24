# SVHN_Dataset_Detection

## Requirements
- `numpy == 1.17`
- `opencv-python >= 4.1`
- `torch == 1.6`
- `torchvision`
- `matplotlib`
- `pandas`
- `numpy`
- `pycocotools`
- `tqdm`
- `pillow`
- `hdf5`
- `PIL`
- `tensorboard >= 1.14`


## File Structure
      .
      ├──FasterRCNN
      |   ├──detection (Reference [2])
      |   |   ├──coco_eval.py
      |   |   ├──coco_utils.py
      |   |   ├──engine.py
      |   |   ├──transforms.py
      |   |   └──utils.py
      |   |
      |   ├──src
      |   |   ├──dataset.py
      |   |   ├──model.py
      |   |   ├──transforms.py
      |   |   └──utils.py
      |   |
      |   ├──00_mat2df.py (Reference [1])
      |   ├──01_train.py  (Reference [3])
      |   ├──02_test.py
      |   ├──train_data.csv
      |   └──valid_data.csv
      |
      └──YOLOv4 (Reference [4])
            ├──cfg
            |   └──yolov4-pacsp.cfg
            |
            ├──data
            |   ├──hyp.scratch.yaml
            |   └──svhn.yaml
            |
            ├──models
            |   ├──export.py
            |   └──svhn.yaml
            |
            ├──utils
            |   ├──activations.py
            |   ├──adabound.py           
            |   ├──autoanchor.py            
            |   ├──datasets.py          
            |   ├──evolve.sh            
            |   ├──gcp.sh           
            |   ├──general.py           
            |   ├──google_utils.py
            |   ├──layers.py
            |   ├──loss.py
            |   ├──metrics.py         
            |   ├──parse_config.py
            |   ├──metrics.py            
            |   ├──plots.py          
            |   ├──torch_utils.py           
            |   └──utils.py 
            |
            ├──generate_submission.py (by myself)
            ├──mat2yolo.py (by myself)
            ├──new_digitStruct.mat (by myself)            
            ├──requirements.txt      
            ├──split_train_valid.py (by myself)            
            ├──test.py
            └──train.py


## Training
## Inference
## Reproducing Submission
To reproduce our submission without retraining, do the following steps

## Results
Faster-RCNN and YOLOv4 achieve the following performance:
| Model    | Faster-RCNN | YOLOv4 |
| -------- | ----------- | ------ |
| mAP      | Text        | Text   |


## Reference
### Faster RCNN
[1] https://github.com/kayoyin/digit-detector/blob/master/construct_data.py  
[2] https://github.com/pytorch/vision/tree/main/references/detection  
[3] https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

### YOLOv4
[4] https://github.com/WongKinYiu/PyTorch_YOLOv4
