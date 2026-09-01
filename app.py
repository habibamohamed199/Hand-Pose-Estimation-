import numpy as  np
import cv2
import mediapipe as mp
import time
import warnings
import sklearn
warnings.filterwarnings("ignore")


import pickle

# load scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

print("Scaler and model loaded successfully!")

mp_drawing = mp.solutions.drawing_utils
hand = mp.solutions.hands
mp_hands = mp.solutions.hands.Hands()

cap=cv2.VideoCapture(0)

while True:
    ret,image=cap.read()
    if ret==False :
        break
    rows, cols, _ = image.shape
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = mp_hands.process(image)
    image.flags.writeable = True
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            swap = []
            x = [ int(landmark.x * cols) for landmark in hand_landmarks.landmark]
            y = [ int(landmark.y * rows) for landmark in hand_landmarks.landmark]
            swap.extend(x)
            swap.extend(y)
            swap = np.array(swap)
            swap = scaler.transform(swap.reshape(1, -1))
            y_pred = model.predict(swap)[0]
            cv2.putText(image, y_pred, (10,50), cv2.FONT_HERSHEY_SIMPLEX,1.2,(255,0,0), 3)
            mp_drawing.draw_landmarks( image, hand_landmarks,hand.HAND_CONNECTIONS)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    

    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()