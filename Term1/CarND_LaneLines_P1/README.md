# **Finding Left and Right Lane Lines on the Road in Images and Videos** 


<img src="examples/laneLines_thirdPass.jpg" width="480" alt="Combined Image" />

Overview
---

When we drive, we use our eyes to decide where to go.  The lines on the road that show us where the lanes are act as our constant reference for where to steer the vehicle.  Naturally, one of the first things we would like to do in developing a self-driving car is to automatically detect lane lines using an algorithm.Ik





The goal of this project is detecting of lane lines in road images and videos using Python and OpenCV.

## Pipeline
These are the steps to detect left and right lanes in a road image:

1- Convert color image to grayscale

2- Apply Gaussian smoothing

3- Edge Detection using Canny

4- Region selection using fillpoly and mask the image

5- Line detectin using Hough transoform on edge detected image

6- Finding left and right lanes (by slope ((y2-y1)/(x2-x1)))

7- Average the position of each of the lines and extrapolate to the top and bottom of the lane.
8- Draw lanes on the original image

9- Save the result (image or video)


To meet specifications in the project, take a look at the requirements in the [project rubric](https://review.udacity.com/#!/rubrics/322/view)

## Test Images
These are the test images that we use to test the pipeline:

![jpg](test_images/solidWhiteCurve.jpg) | ![jpg](test_images/solidWhiteRight.jpg)
