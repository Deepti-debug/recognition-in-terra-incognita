# recognition_in_terra_incognita

## Classification via Inception-v3

**Experiment 1**
- Data distribution:
    - **Train**: train_annotations
    - **Val**: cis_val_annotation
    - **Test**: cis_test_annotations
 - Use the training images as they are to train the model. 

**Experiment 2**
- Data distribution:
    - **Train**: train_annotations
    - **Val**: trans_val_annotation
    - **Test**: trans_test_annotations
 - Use the training images as they are to train the model. 


## Detection via Faster RCNN with ResNet-101 backbone:
- This ReadMe contains high-level info on Detection model, for in-depth information, go to the **Detection-Faster-RCNN** directory and take a look at the ReadMe in that folder. 

**Experiment 1:**
- Data distribution:
    - **Train**: train_annotations
    - **Val**: cis_val_annotations
    - **Test**: cis_test_annotations

- [Install the necessary requirements to train the model](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html#tf-models-install-coco)

- [Train the object detection model on custom data](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html)

- [Monitor training job progress using tensorboard](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#monitor-training-job-progress-using-tensorboard):

- [Export the trained model once training completes](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#exporting-a-trained-model)


**Experiment 2:**
- Data distribution:
    - **Train**: train_annotations
    - **Val**: trans_val_annotations
    - **Test**: trans_test_annotations

- For training and evaluation, use the same steps as **Experiment 1**

## Detection via Faster RCNN with  Inception-Resnet-V2 backbone:

**Experiment 3:**
- Data distribution: same as **Experiment 1**

**Experiment 4:**
- Data distribution: same as **Experiment 2**

## Detection via YOLOv5 single-stage detector:
**Experiment 5:** (Extension we propose)

- Data distribution:
    - **Train**: train_annotations
    - **Val**: trans_val_annotations
    - **Test**: trans_test_annotations

- For training and evaluation, inspiration is drawn from https://github.com/ultralytics/yolov5 

## Domain Generalization Experiments on TerraIncognita:

- The experiments are performed on only a subset of the TerraIncognita Dataset.Images part of 4 locations/domains L100,L38,L43,L46.  The models are trained/validated on 3 domains and tested on the remaining domain. 
- First install virtual environment using DG_env_requirements.txt file.
- First we train the models with ERM_SMA protocol.
- Then we get the final test accuracy using Ensemble of Averages(EoA) for the above trained models.
- run_erm_sma.sh bash script first trains the models using ERM_SMA protocol and then takes the ensemble of these models to give the final test accuracy.
- If you only need the training of the models comment the 3rd and 4th python commands in the script. The 3rd and 4th python commands perform EoA and gets the final test accuracy. 1st and 2nd python commands perform ERM,ERM_SMA protocol for training the models.
- You can find the path to the trained models for each domain using only one trial. The experiments are conducted for more trials but due to space issues only models trained for 1 trail are put on my ADA share3 folder. You can get the test accuracy by running the bash script with the 4th python command by giving out_dir with this trained models directory. The data_dir can also be found on /share3/girmaji08/terra_incognita for these experiments.
- For models with more trails you can get it on gnode026 in the folder /ssd_scratch/cvit/girmaji08/rerm-sma_resnet50 


## Resources

- [Presentation of this work](https://www.canva.com/design/DAFh8VpV6P0/WMOJE6te8_cxWD43tfvt3Q/view?utm_content=DAFh8VpV6P0&utm_campaign=designshare&utm_medium=link&utm_source=publishsharelink)
