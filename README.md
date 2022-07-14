# Sign-Language-Recognition-Using-Mediapipe


## Description how I do
	•	I use classifier and detector as well in the project.
	•	In the project you are able to classify letters (e.g. A, B, C, etc.), sentences (e.g. “I LOVE YOU”, etc.) through signs and actions.
	•	So how I am able to achieve this
	•	First of all, I locate the object which will be our hand, and find the position of it, from there onwards, I am going to classify what exactly is the   hand representing.
	•	So I use the detector from Mediapipe library to detect the hands. 
	•	After detection the next part is classification, so for classification I use classifier from Tensor flow library. 
	•	So I need to write four scripts, the first one is for collecting the data of desire action of class. Whenever I have a hand it supposed to detect it and crop the image and get multiple images of hand, that image will use for training ML algorithm. 
	•	The second script is basically a class for detecting hand.
	•	The third script is for class of classifier from tensor flow.
	•	The fourth script also a class for detecting face mesh.
	•	The fifth script is for testing the hands.

![image](https://user-images.githubusercontent.com/109298390/179020669-34df28af-e317-418c-8726-11581b321768.png)

## Description of files

### Track_Hand.py
	
	This file is basically containing class named as handTracker(). The class handTracker() contain three functions (1. initialization function, 2. findAndDrawHands() function, 3. findLandmarks() function)
	1. initialization function: This function used to initialize the mediapipe library function, which takes the parameters like how many hands you want to detect, how much accurate etc. The first parameter is static mode which is false because i want to detect hands if confidence level is suitable, if put ture it will always do the detection. 
	2. findAndDrawHands() function: This fucntion used to draw the 21 landmarks connected with line, as shown in image below.
	![image](https://user-images.githubusercontent.com/109298390/179025242-11785c82-15e1-48ad-8f26-dbc2b079ea4d.png)

	
