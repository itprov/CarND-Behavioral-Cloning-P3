# **Behavioral Cloning**

## Writeup
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/center_sample.jpg "Center Lane Driving Sample"
[image2]: ./images/recovery_sample1.jpg "Recovery Sample 1"
[image3]: ./images/recovery_sample2.jpg "Recovery Sample 2"
[image4]: ./images/recovery_sample3.jpg "Recovery Sample 3"
[image5]: ./images/normal_sample.jpg "Normal Sample"
[image6]: ./images/flipped_sample.jpg "Flipped Sample"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* setup_data.py containing the script to setup and preprocess data
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* video.mp4 showing video recording of the vehicle driving autonomously one lap around the track
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh python drive.py model.h5```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 6 and 48 (model.py lines 25-28). It includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (line 24). Couple fully connected layers are added at the end (lines 31-32)

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 29, 33).

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 126). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an Adam optimizer, so the learning rate was not tuned manually (model.py line 25). I experimented with various various batch sizes and epochs, and found that the default batch size of 32 and number of epochs = 5 works best.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. I also augmented data using image & angle flipping.

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with known models and experiment with number of convolutional, max pooling, dropout layers as well as number of filters and kernel size.

My first step was to use the LeNet convolution neural network model. As this is a well-known model, I wanted to see how it performs on my data set.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set, with validation set = 20% of total data. I found that my first model had a lower mean squared error on the training set (0.0432) but a higher mean squared error on the validation set (0.0881). This implied that the model was overfitting.

To combat overfitting, I modified the model to add Dropout layers. This resulted in validation loss similar to training loss (both ranging from 0.05 to 0.07).

I wanted to see if I could further reduce validation loss, so I tried various different models, including the autonomous vehicle NVIDIA model mentioned in the "Even more powerful network" video by Udacity. The NVIDIA model worked very well, but took more than 5 minutes for each training epoch. After experimenting with various combinations of Convolutional, MaxPool, Dropout layers, I found an architecture that did not overfit or underfit the dataset, trained faster (~2.25 minutes for each epoch), and did not sacrifice performance significantly.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, e.g. the dirt patches, red & white stripes, the edge walls and the end of the bridge. To improve the driving behavior in these cases, I collected more data for recovery from left and right sides of the road (recording only the recovery portion). I added more samples for the portions of the tracks that continued to make the vehicle run off from the edges.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 22-37) consisted of a convolution neural network with the following layers and layer sizes:

| Layer                 | Description                                   |
|:---------------------:|:---------------------------------------------:|
| Input                 | 160x320x3 RGB image                           |
| Convolution 5x5 + RELU| 2x2 stride, valid padding, outputs 78x158x6 	|
| Convolution 5x5 + RELU| 2x2 stride, valid padding, outputs 37x77x12 	|
| Convolution 5x5 + RELU| 2x2 stride, valid padding, outputs 17x37x24 	|
| Convolution 3x3 + RELU| 1x1 stride, valid padding, outputs 15x35x48	  |
| Dropout               | Dropout rate = 50%                            |
| Flattened             | outputs 25200                                 |
| Fully connected       | outputs 50.                                   |
| Fully connected       | outputs 10.                                   |
| Dropout               | Dropout rate = 50%                            |
| Output                | outputs 1.                                    |
|                       |                                               |



#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded one lap on track one using center lane driving. Here is an example image of center lane driving:

![Center Lane Driving Sample][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to get back to the center of the road in case it veers off toward the edges. These images show what a recovery looks like starting from the end of the bridge and the dirt road :

![Recovery Sample 1][image2]
![Recovery Sample 2][image3]
![Recovery Sample 3][image4]

To augment the data sat, I also flipped images and angles thinking that this would help generalize the data and remove the bias toward left turns. For example, here is an image that has then been flipped:

![Normal Sample][image5]
![Flipped Sample][image6]

I had also used counter-clockwise driving to collect data, but later discarded that data as it did not seem to help much in improving the model performance. Just using the clockwise driving data and flipping each image to augment the data set achieved similar performance.

After the collection process, I had 18104 data points (9052 x 2). I then preprocessed this data by adding a lambda layer that normalizes the data points (model.py line 24).

I finally randomly shuffled the data set (line 111) and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was between 4 and 7 as evidenced by results of multiple experiments (validation loss stops decreasing or reduces very slowly after 4 epochs). I selected 5 as the number of epochs for the final model. I used an adam optimizer so that manually training the learning rate wasn't necessary.

I also used Keras checkpoints mechanism to save the model (only if it's better than before). Between iterations of data collection, I used the load_model function to load the model trained with previous data set, which resulted in faster training and further decrease in validation loss (similar to the transfer learning method.)
