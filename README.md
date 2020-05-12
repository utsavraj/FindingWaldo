# <img src="/README Resources/684929827Find Waldo.jpg" width="150" align="center" > Finding Waldo
[WORK IN PROGRESS]

Using transfer learning to retrain a pre-trained CNN model to find Waldo in a given picture made using TensorFlow Object Detection API.

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

## Training the model

## Testing the model

## Examples
