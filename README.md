# <img src="/README Resources/684929827Find Waldo.jpg" width="150" align="center" > Finding Waldo

Using transfer learning to retrain a pre-trained SSD model to find Waldo in a given picture made using TensorFlow Object Detection API.

**Note** *: This work is no way intended to infringe upon any copyright or trademark.*

## Background
Where's Waldo consist of a series of detailed double-page spread illustrations depicting dozens or more people doing a variety of amusing things at a given location. 

Readers are challenged to find a character named Wally hidden in the group. Wally is identified by his red-and-white-striped shirt, bobble hat, and glasses.

*Source: [Wikipedia](https://en.wikipedia.org/wiki/Where%27s_Wally%3F)*

## Environment Setup
* Python 3.7.5 and Tensorflow version 1.15.2
  * *Numpy version 1.16.2* as Numpy 1.18.0 breaks the evaluation process for both TensorFlow and PyTorch object detection
* [Google Drive](https://www.google.com/drive/) and [Google Collab](https://colab.research.google.com/) is used for this project.

### Google Collab
* Open a new notebook
* From the top left menu: Go to Runtime -> Change runtime type -> select GPU from hardware accelerator.
  * Some pretrain Models support TPU. Ours uses GPU

**Important Note:** The kernel disconnects shortly after your computer sleeps or after using the Colab GPU for 12 hours. Sadly, the training will need to be restarted from scratch if the trained model did not get saved

### Google Drive
* Create a folder named *object_detection* with a subfolder named *data*
* Create two subfolder in the folder *data* named *images* and *annotations*
  * It should look like this:
    ```
    object_detection
               └── data
                     ├── images
                     │
                     └── annotations                         
     ```
* Add the images to *images* folder and the labelled data (XML files ~ explained in Data Preparation) to *annotations* folders respectively.
  * It should look like this:
    ```
    object_detection
               └── data
                     ├── images
                     │      ├── image_1.jpg
                     │      ├── image_2.jpg
                     │      └── ...
                     │
                     └── annotations
                            ├── image_1.xml
                            ├── image_2.xml
                            └── ...                            
     ```

## Data Preparation
* Custom Dataset of Where's Waldo picture was created using Google, Tumblr and other resources.
  * **NOTE:** Images should be in `jpg` format. If you find other formats like `png` etc, convert them using online converters.
  * Located in folder named *object_detection/data/images* 
  * In order to train a strong model:
    * The target (Wally) varies in size (a scaling issue)
    * Images have repeating patterns of differing significance (red & white stripes present on other objects)
    * There is occlusion (where Wally is partially blocked from view by other scene objects)
    * Converting images to black-white and grayscale.
  
* The data was labelled using [LabelImg](https://github.com/tzutalin/labelImg)
  * While labelling data, try to cover using the rectangle all the distinctive features of Waldo.
  * Annotations are saved as XML files in PASCAL VOC format, the format used by ImageNet.
  * To Run LabelImg, use the following command in Terminal for MacOS:
    * `git clone https://github.com/tzutalin/labelImg.git`
    * `cd labelImg`
    * `pip3 install pyqt5 lxml` 
    * `make qt5py3`
    * `python3 labelImg.py`
    
       <img src="README Resources/labelimg.gif" width="560" />

## Data Preprocressing

* Dividing the `xml` files into test and train labels.
  * **NOTE**: They will be moved to *test_labels* and *train_labels* by the hash value. 
  * It should look like this:
   ```
    object_detection
               └── data
                     ├── images
                     │      ├── image_1.jpg
                     │      ├── image_2.jpg
                     ├── annotations/
                     │      └── [EMPTY]
                     ├── train_labels/
                     │      ├── image_1.xml
                     │      ├── image_4.xml
                     │      └── ...
                     └── test_labels/
                            ├── image_2.xml
                            ├── image_3.xml
                            └── ...                   
     ```
* Create two `csv` files for the `.xml` files in each *train_labels* and *test_labels* as well a `pbtxt` file.
  * These two csv files will contain each image’s file name, the label, box position, etc. 
  * Also, more than one row is created for the same picture if there is more than one class or label for it in a csv file.
  * `pbtxt` file that will contain the label map for each class. This file will tell the model what each object is by defining a mapping of class names to class ID numbers
  * It should look like this:
   ```
    object_detection
               └── data
                     ├── images
                     │      ├── image_1.jpg
                     │      ├── image_2.jpg
                     ├── annotations/
                     │      └── [EMPTY]
                     ├── train_labels/
                     │      ├── image_1.xml
                     │      ├── image_4.xml
                     │      └── ...
                     ├── test_labels/
                     │      ├── image_2.xml
                     │      ├── image_3.xml
                     │      └── ... 
                     ├── label_map.pbtxt
                     │
                     ├── test_labels.csv
                     │
                     └── train_labels.csv
     ```
     
* Tensorflow accepts the data as tfrecords (which is a binary file that run fast with low memory usage), hence packing our labels (saved as a `.csv`) and images (`.jpeg`) into a single binary `.tfrecord` file. 
  * Add your custom object text in the function class_text_to_int below by changing the row_label variable (This is the text that will appear on the detected object). *Add more labels if you have more than one object.
  eg. 
     ```
     def class_text_to_int(row_label):
      if row_label == 'NAME_OF_THE_LABEL':
        return 1
      else:
        None
     ```
  * It should look like this:
  ```
    object_detection/
               └── data/
                    ├── images/
                    │      └── ...
                    ├── annotations/
                    │      └── [EMPTY]
                    ├── train_labels/
                    │      └── ...
                    ├── test_labels/
                    │      └── ...
                    ├── label_map.pbtxt
                    ├── test_labels.csv
                    ├── train_labels.csv
                    ├── test_labels.records
                    └── train_labels.records
   ```
   
## Model
* Installing TensorFlow Object Detection API Models using https://github.com/tensorflow/models.git
  * The new *model* folder now should have a pretrained model with a `.ckpt` checkpoint file
  
* While the model could be trained from scratch starting with randomly initialised network weights, this process would probably take weeks. Hence, we used a pretrain model on the COCO dataset using transfer learning to understand our label by training from our dataset.

* We are using Single Shot MultiBox Detector, specifically **ssd_mobilenet_v2_coco** model.
  * ***Version***: *ssd_mobilenet_v2_coco_2018_03_29*
  * More information about SSD and other models: https://heartbeat.fritz.ai/a-2019-guide-to-object-detection-9509987954c3#b49b
* Check that all labels are present in *labels.txt* so it looks for that object and labels it.
  eg.
     ```
     item {
     id: 1
     name: 'waldo'
     }
     ```
* It should look like this:
   ```
   object_detection/
              ├── data/
              │    ├── images/
              │    │      └── ...
              │    ├── annotations/
              │    │      └── ...
              │    ├── train_labels/
              │    │      └── ...
              │    ├── test_labels/
              │    │      └── ...
              │    ├── label_map.pbtxt
              │    ├── test_labels.csv
              │    ├── train_labels.csv
              │    ├── test_labels.records
              │    └── train_labels.records
              │
              └── models/           
                   ├── research/
                   │      ├── training/
                   │      │      └── ...
                   │      ├── pretrained_model/

                   │      ├── frozen_inference_graph.pb
                   │      └── ...
                   └── ...
    ```
## Configuring Training Pipeline
* Configuration files (`.config`) are just Protocol Buffers objects described in the `.proto` files under *object_detection/models/research/object_detection/protos*. 
* They need to be edited in order to run the model. The sample config files are located *object_detection/models/research/object_detection/samples/configs/*
* We use `ssd_mobilenet_v2_coco.config` 
  * The required edit includes:
    * In `model {} > ssd {}` change `num_classes` to the number of classes/labels
   ```
    # For only one class (eg. just Waldo)
    model {
     ssd {
       num_classes: 1
   ```
    * `train_config {}`: change fine_tune_checkpoint to the checkpoint file path. **NOTE:** The exact file name model.ckpt doesn't exist. This is where the model will be saved during training.
    
   ` fine_tune_checkpoint: "/gdrive/My Drive/object_detection/models/research/pretrained_model/model.ckpt" `
   
    * `train_input_reader {}`: set the path to the `train_labels.record` and the label map `pbtxt` file.
    
      ```
      train_input_reader: {
        tf_record_input_reader {
          #path to the training TFRecord
            input_path: "/gdrive/My Drive/object_detection/data/train_labels.record"
        }
           label_map_path: "/gdrive/My Drive/object_detection/data/label_map.pbtxt"
       }
      ```
    * `eval_input_reader {}`: set the path to the `test_labels.record` and the label map `pbtxt` file.
    
         ```    
         eval_input_reader: {
           tf_record_input_reader {
             #path to the testing TFRecord
             input_path: "/gdrive/My Drive/object_detection/data/test_labels.record"
           }
           #path to the label map file
           label_map_path: "/gdrive/My Drive/object_detection/data/label_map.pbtxt"
           shuffle: false
           num_readers: 1
         }
         ``` 
### Optional Edits to Config file    
* In `train_config {}`, you can add image augmentation. List can be found here: 

     ```
     data_augmentation_options {
         random_adjust_contrast {
         }
       }
       data_augmentation_options {
         random_rgb_to_gray {
         }
       }
       data_augmentation_options {
         random_vertical_flip {
         }
       }
       data_augmentation_options {
         random_rotation90 {
         }
       }
       data_augmentation_options {
         random_patch_gaussian {
         }
     ```
* In `model {} > ssd {} > box_predictor {}`: set `use_dropout` to `true` This will be helpful to counter overfitting.

* In `eval_config : {}` set the number of testing images you have in `num_examples` and remove `max_eval` to evaluate indefinitely

       ```
       eval_config: {
         num_examples: 400 # the number of testing images
         num_visualizations: 20 # the number of visualization to see in tensorboard
       }
       ```

## Tensorboard
* While the training is running, you can check the accuracy of the model with [Tensorboard](https://www.tensorflow.org/tensorboard/) (implemented using [ngrok](https://ngrok.com/) - go to the link it provides)
* Using tensorboard,  monitor the loss, mAP, AR, the pictures and the annotations during training. At each evaluation step, you could see how good your model was at detecting the object. 
   * **NOTE**: A max of 20 connection per minute is allowed when using ngrok, you will not be able to access tensorboard while the model is logging.

<img src="/README Resources/TensorBoard.png" width="900" align="center" >


## Training the Model
* Go to: *object_detection/models/research/object_detection*
* Run command `python3 object_detection/model_main.py \ --pipeline_config_path= PATH_TO_PIPELINE_CONFIG --train_dir=PATH_TO_TRAIN_DIR`
  * `model_main.py` which runs the training process
  * `PATH_TO_PIPELINE_CONFIG` is the path to our pipeline config file. eg. `--pipeline_config_path=/gdrive/My\ Drive/object_detection/models/research/object_detection/samples/configs/ssd_mobilenet_v2_coco.config`
  * `PATH_TO_TRAIN_DIR` is a newly created directory where our new checkpoints and model will be stored. `--model_dir=training/`
  
* Look for **loss** in the output.
   <img src="/README Resources/output_loss.png" width="150" align="center" >
  * Loss is a summation of the errors made for each example in training or validation sets. The lower, the better - if it’s slowly decreasing, that means that your model is learning (…or overfitting your training data).
* The script will automatically store a checkpoint file after a certain number of steps (600 seconds or 5 steps), so that you can restore your saved checkpoints at any time in case your computer crashes while learning. This means that when you want to finish training the model, you can just terminate the script
* The general rule as to when to stop training is when the loss on our evaluation set stops decreasing or is generally very low.

## Testing the Model

## Examples


