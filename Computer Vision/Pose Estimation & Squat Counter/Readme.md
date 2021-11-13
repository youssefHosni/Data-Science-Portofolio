## Pose Estimation & Squat Counter

### Introdction 

Pose estimation refers to a general problem in computer vision techniques that detect human figures in images and videos, so that one could determine, for example, where someone’s elbow shows up in an image. It is important to be aware of the fact that pose estimation merely estimates where key body joints are and does not recognize who is in an image or video. The pose estimation models takes a processed camera image as the input and outputs information about keypoints. The keypoints detected are indexed by a part ID, with a confidence score between 0.0 and 1.0. The confidence score indicates the probability that a keypoint exists in that position. An example of this is as shown in the video below.

![alt-text](https://github.com/youssefHosni/Data-Science-Portofolio/blob/main/Computer%20Vision/Pose%20Estimation%20%26%20Squat%20Counter/jump.gif)

Based on the results of the pose estimatiaon, the squat movment was detected and counted and printed on the screen as shown in the video below.

---

### Methods 

The model used is MoveNet, the MoveNet is available in two flavors:

* MoveNet.Lightning is smaller, faster but less accurate than the Thunder version. It can run in realtime on modern smartphones.
* MoveNet.Thunder is the more accurate version but also larger and slower than Lightning. It is useful for the use cases that require higher accuracy.

MoveNet.Lightning is used here.

MoveNet is the state-of-the-art pose estimation model that can detect these 17 key-points:
 
* Nose
* Left and right eye
* Left and right ear
* Left and right shoulder
* Left and right elbow
* Left and right wrist
* Left and right hip
* Left and right knee
* Left and right ankle

The various body joints detected by the pose estimation model are tabulated below:

| Id	| Part |
| --- | ----------- |
| 0	 | nose |
| 1	| leftEye |
| 2	| rightEye |
| 3	| leftEar |
| 4	| rightEar |
|5	| leftShoulder |
| 6 	| rightShoulder  |
| 7	| leftElbow |
| 8	| rightElbow |
| 9	| leftWrist |
| 10 | rightWrist |
| 11	| leftHip |
| 12	| rightHip |
| 13	| leftKnee |
| 14	| rightKnee |
| 15	| leftAnkle |  
| 16	| rightAnkle |


---

### Install dependencies

"pip install -r requirements"

