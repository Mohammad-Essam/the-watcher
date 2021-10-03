
# Datum2.0
# The Team
### Team Members
* Team Leader :  Mohammad Essam Mohammad
* Team Member 1 : Bassem Essam Mohammad


# Toy Project
## Problem Statement
Counting the number of edges that the shapes contain using contours. except for the circle

## Learning Process
in toy project part, we learned about OpenCV, contours,  and how to manipulate images with it.


 
# Coding Competition
## Problem Statement
Monitoring cams and software are good, but it needs someone to watch the camera the entire time, to ensure that everything is running as it should be.
but what if when no one is watching something happened, maybe the baby is making his way to the kitchen. maybe your cat is on the dining table eating your food.

maybe you don't want anyone to park in a non-parking area, so you want to be notified if any car approach this area.

so we bulit "The Watcher", it will always watch for you.
## Solution

*   The application gives the user the freedom to set parts of the surroundings as a forbidden region(let's call it a danger zone) for some objects.
so if the objects touch that region the application will alert the user with some alerting sound.
and  each class of object has a different setup and different regions
**i.e** for electronic devices objects may have the class of bottle of water as a danger zone, so if any bottle came near any electronic devices' range the application will alert the user.
* Frameworks/Tools/Technologies stacks used:
 **Opencv**: for manipulating the frames of the video/webcam and for loading the model that detects objects
 **PyQt5**: for the Graphical user interface (GUI)
**SSD_Mobilenet model**: a caffe implementation of SSD Object Detection model with a mobilenet backbone  for detecting the objects from the camera
* why I choose those technologies:
 **Opencv** is easy to use. fast and robust
**PyQt** works well with python and gives a native look and feel for the widget. so the application won't  look strange 
**SSD_Mobilenet model**: because it's really fast and could work in real-time with a good frame rate.
**Constraints of the used technologies**:
the pretrained model we use. is trained on  20 classes only. 
in contrary to pytoch ssdlite320_mobilenet_v3_large model that contains 91 different class. 
but the issue with pytorch is that it's heavier on our machine.
## Methodology 

  * We used a pretrained SSD approach model with Mobilenet backbone, for detecting the objects.
  here it's link 
  https://github.com/chuanqi305/MobileNet-SSD

## System Architecture proposal (if any)
- The application uses the camera/ webcam to detect the objects, and takes the user inputs through mouse interactions, via drawing danger regions for each intended object

## Demo Video
https://drive.google.com/drive/folders/1g4psWpqdd2xO1SKFrXoxFuo2AKJvoUj_?usp=sharing

## Screenshots
* ### How to use the application
* 1- here is the application interface, you should choose some object to 'Watch', to be able to specify its danger zones.
* 2- then you press the button Watch. you can watch for multiple different classes of objects and each object will have its own rule.
![img](https://github.com/MohammadEssam0/the-watcher/blob/page/docs/hack1.jpg)
- 3- after pressing Watch, the name of the class of objects will be in the combo box.
![img](https://github.com/MohammadEssam0/the-watcher/blob/page/docs/hack2.jpg)
- 4- now you can specify the danger zone by hand **(static regions method)** ,first choose the object from the combobox, then draw a square on the screen such as in this image.(here we draw a danger zone for the dining table, and while the dining table  is within the danger zone, then the apllication says 'danger exists and make an alerting sound'), you can draw multiple danger zones for an object.
![img](https://github.com/MohammadEssam0/the-watcher/blob/page/docs/hack3.jpg)
- Or by assigning another object to the current chosen object(the one in the combo box) via the button 'Add danger object' **(let's call this method a dynamic regions method)**. by pressing it another popup will appear, you should choose a class of objects, and that class of objects will be treated as a danger zone for your intended object. as the following image. here we assign a danger object for the dining table class of objects.
![img](https://github.com/MohammadEssam0/the-watcher/blob/page/docs/hack4.jpg)

## Steps to run the software
- download and install python3 
- for easy installing the packages you may install pip first
### install opencv
- pip3 install opencv-python
### install PyQt5
- pip3 install PyQt5
### running the application
- python3 thewatcher.py
