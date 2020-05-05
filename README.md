# <img src="/README Resources/684929827Find Waldo.jpg" width="150" align="center" > Finding Waldo
[WORK IN PROGRESS]

A custom object detection model to find Waldo in a given picture made using TensorFlow Object Detection API.

**Note** *: This work is no way intended to infringe upon any copyright or trademark.*

## Background
Where's Waldo consist of a series of detailed double-page spread illustrations depicting dozens or more people doing a variety of amusing things at a given location. 

Readers are challenged to find a character named Wally hidden in the group. Wally is identified by his red-and-white-striped shirt, bobble hat, and glasses.

*Source: [Wikipedia](https://en.wikipedia.org/wiki/Where%27s_Wally%3F)*

## Environment Setup
* [Google Drive](https://www.google.com/drive/) and [Google Collab](https://colab.research.google.com/) is used for this project.

### Google Collab
* Open a new notebook
* From the top left menu: Go to Runtime -> Change runtime type -> select GPU from hardware accelerator.
  * Some pretrain Models support TPU. Ours uses GPU

**Important Note:** The kernel disconnects shortly after your computer sleeps or after using the Colab GPU for 12 hours. Sadly, the training will need to be restarted from scratch if the trained model did not get saved

### Google Drive

## Data Preparation
* Custom Dataset of Where's Waldo picture was created using Google, Tumblr and other resources.
  * Located in folder named *UnlabelledData*
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

## Setting up Model

## Training the model

## Testing the model

## Examples
