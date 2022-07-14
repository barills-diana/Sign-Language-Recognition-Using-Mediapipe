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
