# Sign-Language-Recognition-Using-Mediapipe


## Description how I do
	- •	I use classifier and detector as well in the project.
	  •	In the project you are able to classify letters (e.g. A, B, C, etc.), sentences (e.g. “I LOVE YOU”, etc.) through 
	  signs and actions.
	  •	So how I am able to achieve this
	  •	First of all, I locate the object which will be our hand, and find the position of it, from there onwards,
	  I am going to classify what exactly is the   hand representing.
	  •	So I use the detector from Mediapipe library to detect the hands. 
	  •	After detection the next part is classification, so for classification I use classifier from Tensor flow library. 
	  •	So I need to write four scripts, the first one is for collecting the data of desire action of class. Whenever I 
	  have a hand it supposed to detect it and crop the image and get multiple images of hand, that image will use for training
	  ML algorithm. 
	  •	The second script is basically a class for detecting hand.
	  •	The third script is for class of classifier from tensor flow.
	  •	The fourth script also a class for detecting face mesh.
	  •	The fifth script is for testing the hands.

![image](https://user-images.githubusercontent.com/109298390/179020669-34df28af-e317-418c-8726-11581b321768.png)

## Description of files

### Track_Hand.py
	
	- This file is basically containing class named as handTracker(). The class handTracker() contain three functions (1. 
	initialization function, 2. findAndDrawHands() function, 3. findLandmarks() function)
		1. initialization function: This function used to initialize the mediapipe library function, which takes the 
		parameters like how many hands you want to detect, how much accurate etc. The first parameter is static mode 
		which is false because i want to detect hands if confidence level is  suitable, if put ture it will always do 
		the detection. 
		2. findAndDrawHands() function: This fucntion used to draw the 21 landmarks connected with line, as shown in 
		image below.
	
![image](https://user-images.githubusercontent.com/109298390/179025242-11785c82-15e1-48ad-8f26-dbc2b079ea4d.png)
		
		3. findLandmarks() function: This function return the landmarks id, x-axis, y-axis, and bounding box for hand.

### FaceMeash.py

	- This file is basically containing class named as FaceMesh(). The class FaceMesh() contain three functions (1. 
	initialization function, 2. drawFaceMesh() function, 3. meshLandmarks() function)
		1. initialization function: This function used to initialize the mediapipe library function, which takes the 
		parameters like how many face you want to detect, how much accurate etc. 
		2. drawFaceMesh() function: This fucntion used to draw the 468 landmarks connected with line, as shown in 
		image below.
		
![image](https://user-images.githubusercontent.com/109298390/179029128-f165bcdf-a68f-41d0-b6f2-57119ce62a38.png)

		3. meshLandmarks() function: This function return the landmarks id, x-axis, y-axis, and bounding box for face.

### CollectingData.py

	- This file contain the code to collect the dataset of desire action, letter or sign using mediapipe and opencv. I collect
	the data for the following letters and signs ("A", "D", "GOODBYE", "HELLO", "I", "I LOVE YOU", "M", "N", "NO", "PLEASE", 
	"SORRY", "WELCOME", "YES"). 
	- This is generic code to accuire the desire data, we can collect data for any kind of sign or letter and train the model.
	- For training the desire data i use Google Teachable Machine website, the link is mentioned below. 
	https://teachablemachine.withgoogle.com/ 

### Classification.py

	- This file contain class Classifier() and function getPrediction(), which uses tensorflow keras model to classify and predict 
	the class of specific an object.

### TestingClasses.py

	- This is the main code file wich is used to test the trained dataset with run time webcam feed. 
	_ It uses handTracker() class to draw and track the hand, FaceMesh() class to draw and track face, TensorFlow keras classifier to 
	classify and predict the class. 
	
### keras_model.h5

	_ Trained Keras model file from Google Teachable Machine. Input to the tensorflow classifier class.

### labels.txt

	- Containing the classes name.
