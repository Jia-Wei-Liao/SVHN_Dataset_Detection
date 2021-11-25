# SVHN_Dataset_Detection


## Getting the code
You can download all the files in this repository by cloning this repository:
```
https://github.com/Jia-Wei-Liao/SVHN_Dataset_Detection.git
```


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
          ├──checkpoint
          |   └── demo_0.419pt (download from Goole Drive)
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


## Training
To train the model, run this command:
```
python train.py --data svhn.yaml --cfg cfg/yolov4-pacsp.cfg --weights checkpoint/yolov4.weights --device 0 --img 640 640 --batch-size 16
```


## Inference
You can download the weight on the Google Drive:  
<https://drive.google.com/drive/folders/1BPxTCnvXPHck3hg5QOFD1xJlMDZplKfh?usp=sharing>  

To inference the results, you should move to folder YOLOv4 and run this command:
```
python generate_submission.py --data_path /test --weight /checkpoint/demo_0.419.pt
```


## Reproducing submission
To reproduce our submission, please do the following steps:
1. Getting the code
2. Install the package
3. Download the dataset
4. Download pre-trained weight
5. Inference


## Results
Faster-RCNN and YOLOv4 achieve the following performance:
| Model                     | Faster-RCNN | YOLOv4   |
| ------------------------- | ----------- | ---------|
| best epoch                | 4           | 50       |
| mAP                       | 0.389141    | 0.41987  |
| speed on P100 GPU (img/s) | X           | 0.13696  |
| speed on K80  GPU(img/s)  | X           | 0.13696  |

You can open our Google Colab on this link:  
<https://colab.research.google.com/drive/1iosQjMUfzmDVLkrXhI13IZyuIJuTiArq?usp=sharing>


## Reference
### Faster RCNN
[1] https://github.com/kayoyin/digit-detector/blob/master/construct_data.py  
[2] https://github.com/pytorch/vision/tree/main/references/detection  
[3] https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

### YOLOv4
[4] https://github.com/WongKinYiu/PyTorch_YOLOv4
