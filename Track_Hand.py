import mediapipe as mp
import cv2

class handTracker():

    # initialization Function
    def __init__(self, Mode = False, maximumHands = 2, modelComplexity = 1,
                 detConfidence = 0.5, trackConfidence = 0.5):
        self.Mode = Mode
        self.maximumHands = maximumHands
        self.modelComplexity = modelComplexity
        self.detConfidence = detConfidence
        self.trackConfidence = trackConfidence

        # so first thing is that we have to create an object from over class hands (this class is from mediapipe lib)
        self.HandsSol = mp.solutions.hands

        self.hands = self.HandsSol.Hands(self.Mode, self.maximumHands,self.modelComplexity,
                                         self.detConfidence, self.trackConfidence)
        # to draw the line between the landmarks mediapipe also provide solution for that
        self.drawLine = mp.solutions.drawing_utils

    def findAndDrawHands(self, frame):

        RGBimage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        self.outCome = self.hands.process(RGBimage)

        if self.outCome.multi_hand_landmarks:
            for handLandmarks in self.outCome.multi_hand_landmarks:
                self.drawLine.draw_landmarks(frame, handLandmarks,
                                             self.HandsSol.HAND_CONNECTIONS)

        return frame

    def findLandmarks(self, frame, handNo = 0):

        landMarksList = []
        x_list = []
        y_list = []
        bbox = []

        if self.outCome.multi_hand_landmarks:
            myHand = self.outCome.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):  # enumerate returns both id and landmarks
                # print(id)
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id, cx, cy)
                x_list.append(cx)
                y_list.append(cy)
                landMarksList.append([id, cx, cy])

                ## bbox
                xmin, xmax = min(x_list), max(x_list)
                ymin, ymax = min(y_list), max(y_list)
                boxW, boxH = xmax - xmin, ymax - ymin
                bbox = xmin, ymin, boxW, boxH


        return landMarksList, bbox
