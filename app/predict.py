import pickle
import mediapipe as mp
import numpy as np

loaded_dict = pickle.load(open('app/model.p', 'rb'))
hand_model = loaded_dict['model']

def predict_img(img_rgb):
    
    media_hands = mp.solutions.hands
    
    hand_detector = media_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
    temp_data = []
    x_coordinates = []
    y_coordinates = []

    detection_results = hand_detector.process(img_rgb)
    if len(detection_results.multi_hand_landmarks) != 1:
        return -1,0
    
    landmark_set = detection_results.multi_hand_landmarks[0]
    
    for i in range(len(landmark_set.landmark)):
        x_val = landmark_set.landmark[i].x
        y_val = landmark_set.landmark[i].y

        x_coordinates.append(x_val)
        y_coordinates.append(y_val)
        
    #Normalize x and y
    for i in range(len(landmark_set.landmark)):
        x_val = landmark_set.landmark[i].x
        y_val = landmark_set.landmark[i].y
        temp_data.append(x_val - min(x_coordinates))
        temp_data.append(y_val - min(y_coordinates))
    
    gesture_prediction = hand_model.predict([np.asarray(temp_data)])
    prob = hand_model.predict_proba([np.asarray(temp_data)])
    return gesture_prediction[0], np.max(prob)

