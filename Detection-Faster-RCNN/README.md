# recognition_in_terra_incognita

## Detection Experiments Perfomed
![summary_img](./resouce_imgs/detection-exp-summary.png)

### Experiment 1:
- Model: Faster RCNN with ResNet-101 backbone
- Data distribution:
    - **Train**: train_annotations
    - **Val**: cis_val_annotations
    - **Test**: cis_test_annotations
- Motive of experiment: To see, how well the model performs on similar data distribution

- Root folder:   `exp_1_detect_cis_val_resnet-101`

### Store Images:
- All the images of train, val and test distribution that we will use to train and evaluate this model are stored in the `images`folder of root directory.

#### Create Label Map:
- TensorFlow requires a label map, which namely maps each of the used labels to an integer values. This label map is used both by the training and detection processes.
- The label map for our 15 classes can be found in `./annotations/label_map.pbtxt` of root directory.

#### Create TensorFlow Records:
- Now, we need to convert our annotations into the TFRecord format.
- To convert training images to TFRecord:
    ```
    python generate_tfrecord.py -x ./images/train_annotations -l ./annotations/label_map.pbtxt -o ./annotations/train_annotations.record
    ```
- To convert validation images to TFRecord:
    ```
    python generate_tfrecord.py -x ./images/cis_val_annotations -l ./annotations/label_map.pbtxt -o ./annotations/cis_val_annotations.record
    ```    

#### 

