# **Traffic Sign Recognition**

## Report
###By Roy Veshovda
---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./report/visualization.png "Visualization"
[image11]: ./external-images/ext1.png "Traffic Sign 1"
[image12]: ./external-images/ext2.png "Traffic Sign 2"
[image13]: ./external-images/ext3.png "Traffic Sign 3"
[image14]: ./external-images/ext4.png "Traffic Sign 4"
[image15]: ./external-images/ext5.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

You're reading it! and here is a link to my [project code](https://github.com/royveshovda/CarND-Traffic-Sign-Classifier-P2/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

The code for this step is contained in the 3rd and 4th code cells of the Jupyter notebook.  

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### Details
The code for this step is contained in the 4th code cell of the Jupyter notebook.

First I show a sample of an image from each of the 43 classes.
Below those I show a bar chart of how many images of each class exists in the training data.

That bar chart is also shown below.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Preprocessing

The code for this step is contained in the 5th code cell of the Jupyter notebook.

My intuition was to keep the color channels, as I believe the signs will be easier to separate if colors are also processed by the network.

But I wanted to normalize the color channels between -1.0 and 1.0 (instead of the original 0-255).

I also am running a shuffle of the training data.

#### 2. Training, validation and testing data.

The data came ready split into train, validation and test arrays, so no more pre-processing was needed here.

#### 3. Model architecture

The code for my final model is located in the 6th cell of the Jupyter notebook.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| 1: Convolution 5x5     	| 1x1 stride, same padding, outputs 32x32x64 	|
| 1: RELU					|												|
| 1: Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| 2: Convolution 5x5	    | 1x1 stride, same padding, outputs 16x16x128 	|
| 2: RELU					|												|
| 2: Max pooling	      	| 2x2 stride,  outputs 8x8x128 				|
| 2: Flatten					|												|
| 3: Fully connected		| Input: 8192, output: 4096 	|
| 3: RELU					|												|
| 3: Dropout					|												|
| 4: Fully connected		| Input: 4096, output: 1024 	|
| 4: RELU					|												|
| 4: Dropout					|												|
| 5: Fully connected		| Input: 1024, output: 43 	|



#### 4. Hyperparameters

The code for training the model is located in the 7th and 9th cell of the Jupyter notebook.

I decided to train the model using the optimizer suggested in the LeNet example: AdamOptimizer.
In the end I used 600 epochs and batch size of 1024.

For learning rate I tested a few options, but ended up using 0.001.


#### Approach taken for finding a solution

The code for calculating the accuracy of the model is located in the 12th cell of the Jupyter notebook.

My final model results were:
* validation set accuracy of 97.9%
* test set accuracy of 96.8%

My approach contained many iterations. Here are the most important listed:
* At first I tried plain LeNet network without any normalization or regularization of any kind. That resulted in a poor performance of around 80% accuracy.
* Next step was to normalize the data, but still not real improvements.
* As a last step I introduced dropout, and set the dropout rate to 0.9. This did not give too much improvements either.
* I extended the network to include more nodes at each level, and was able to increase the accuracy to about 95% validation accuracy, but only 91% testing accuracy.
* I decided to increase the number of epochs to 600 and also set the dropout to 0.5



### Test a Model on New Images

Here are five German traffic signs that I found on the web:

![alt text][image11] ![alt text][image12] ![alt text][image13]
![alt text][image14] ![alt text][image15]

My intuition was that image 3 and 5 could be difficult to classify due to the fact that they are skewed a bit and image 3 is also slightly rotated.

#### Model's predictions on these new traffic signs

The code for making predictions on my final model is located in the 14th, 15th and 16th cells of the Jupyter notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Traffic signals      		| Traffic signals 	|
| Ahead only     			| Ahead only	|
| Speed limit (70km/h)					| Yield											|
| Road narrows on the right	      		| Bumpy Road					 				|
| Priority road			| Priority road	|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. This is not too bad considering I have not trained the model with any kind of skewed images.

#### 3. How certain the model?

The code for making predictions on my final model is located in the 17th cell of the Jupyter notebook.

##### Image 1
The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .60         			| Stop sign   									|
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|

##### Image 2
The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .60         			| Stop sign   									|
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|

##### Image 3
The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .60         			| Stop sign   									|
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|

##### Image 4
The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .60         			| Stop sign   									|
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|

##### Image 5
The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .60         			| Stop sign   									|
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|
