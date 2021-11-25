# ENPH 353 Lab Notebook

Xander Naumenko and Nader Kamali

* October 25-31
  * Setup enviornment
    * Got folder structure setup in same way as described in notes
    * Created a launch file for our controller
  * Made new controller and uploaded it to github
    * Ensured publishers/subscribers worked
    * Got robot moving in a circle to test everything worked together
* November 1-7
  * Worked on PID control
    * Only needed to use P to get functional self driving
    * Got robot driving around the track consistently while keeping at least two wheels inside the road
    * Thresholded the road from the subscriber image and found the centroid of the road to use for the proportional response
    * If I and D terms are needed later they should be easy to add
* November 8-14
  * License plate detection
    * Looked into ways of taking an input image and outputing an image containing just the license plate contained in it
    * Tried SIFT
      * Didn't work since the license plates didn't have consistent features to track based off of
    * Since SIFT didn't work we tried thresholding and manually find outside points of license plates
      * Realized that this solution was not ideal and was complicated to implement ourselves
    * Switched to opencv's getContours to find the outside edges of thresholded image and perspectiveTransform to transform it into a nice image
    * https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
* November 15-21
  * Created a branch for time trials
    * Set up timer to start and stop
    * Got the pid running around the track
* November 22-28
  * Added a second threshold to capture licenses on the other side of the track
    * Threshold with the greatest mass (non-zero values) is deemed to be the valid one
  * Trained CNN model on generated license plate images
    * Blurred images by resizing them, imitating the blur of captured licenses
  * Implemented a plate parser that splits the captured license into characters, then reads them using the CNN
  * Fixed Keras versioning issues (pending)
    * https://www.icode9.com/content-4-1153066.html
  * Added state machine to keep track of what state the controller is in at every point
  * Implemented pedestrian dodging using background subtraction
    * https://www.geeksforgeeks.org/background-subtraction-opencv/