# recognition_in_terra_incognita

### Detection via Faster RCNN with ResNet-101 backbone:

**Experiment 1:**

- Data distribution:
    - **Train**: train_annotations
    - **Val**: cis_val_annotations
    - **Test**: cis_test_annotations

- [Install the necessary requirements to train the model](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html#tf-models-install-coco)

- [Train the object detection model on custom data](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html)

- Monitor training job progress using tensorboard <br />
    `tensorboard --logdir=models/my_faster_rcnn_resnet101_v1_1024x1024_coco17_tpu-8`

- [Export the trained model once training completes](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#exporting-a-trained-model)


**Experiment 2:**

- Data distribution:
    - **Train**: train_annotations
    - **Val**: trans_val_annotations
    - **Test**: trans_test_annotations

- For training and evaluation, use the same steps as **Experiment 1**

### Detection via Faster RCNN with  Inception-Resnet-V2 backbone:

**Experiment 3:**

- Data distribution: same as **Experiment 1**

**Experiment 4:**

- Data distribution: same as **Experiment 2**

## 🔗 Detection Resources
- [Train Tensorflow COCO API](https://www.youtube.com/watch?v=XoMiveY_1Z4&t=102s&ab_channel=KrishNaik) 
- [Tensorflow 2.0 model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)
