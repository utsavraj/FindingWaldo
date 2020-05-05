# <img src="/README Resources/684929827Find Waldo.jpg" width="150" align="center" > Finding Waldo
A custom object detection model to find Waldo in a given picture made using TensorFlow Object Detection API.

**Note** *: This work is no way intended to infringe upon any copyright or trademark.*

## Background
Where's Waldo consist of a series of detailed double-page spread illustrations depicting dozens or more people doing a variety of amusing things at a given location. 

Readers are challenged to find a character named Wally hidden in the group. Wally is identified by his red-and-white-striped shirt, bobble hat, and glasses.

*Source: [Wikipedia](https://en.wikipedia.org/wiki/Where%27s_Wally%3F)*

## Data Preparation
* Custom Dataset of Where's Waldo picture was found using Google, Tumblr and other resources.
  * Located in folder named *UnlabelledData*
* The data was labelled using [LabelImg](https://github.com/tzutalin/labelImg)
  * While labelling data, try to cover using the rectangle all the distinctive features of Waldo.
  * Annotations are saved as XML files in PASCAL VOC format, the format used by ImageNet.
  * To Run LabelImg, use the following command in Terminal for MacOS:
    * `git clone https://github.com/tzutalin/labelImg.git`
    * `cd labelImg`

## Data Preprocressing

## Setting up Model

## Training the model

## Testing the model

## Examples
