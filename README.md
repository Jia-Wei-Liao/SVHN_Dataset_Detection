# SVHN_Dataset_Detection

## Requirements
- `numpy == 1.17`
- `opencv-python >= 4.1`
- `torch == 1.6`
- `torchvision`
- `matplotlib`
- `pandas`
- `numpy`
- `scipy`
- `pycocotools`
- `tqdm`
- `pillow`
- `hdf5`
- `PIL`
- `tensorboard >= 1.14`


## File Structure
      .
      ├──FasterRCNN
      |   ├──checkpoint
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
      |   ├──train
      |   |   ├──X.png (30000 pictures)
      |   |   └──digitStruct.mat
      |   |      
      |   ├──test
      |   |   └──X.png (13068 pictures)
      |   |      
      |   ├──00_mat2df.py (Reference [1])
      |   ├──01_train.py (Reference [3])
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
          ├──weights
          |   └──
          |
          ├──checkpoint
          |   └──
          |
          ├──train
          |   └──X.png (30000 pictures)
          |
          ├──test
          |   └──X.png (13068 pictures)        
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
          ├──new_digitStruct.mat (we modify the mat file by MATLAB that scipy package can import)            
          ├──requirements.txt
          ├──split_train_valid.py (by myself)            
          ├──test.py 
          └──train.py


## Training

## Inference
To inference the results, run this command:
```
python generate_submission.py --data_path test -- checkpoint best.pt
```


## Reproducing Submission
To reproduce our submission without retraining, do the following steps

\href{https://drive.google.com/drive/folders/1BPxTCnvXPHck3hg5QOFD1xJlMDZplKfh?usp=sharing}

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
