from flask import Flask, render_template, request, Response
from werkzeug.utils import secure_filename
import joblib
import pandas as pd
from your_image_processing_module import extract_features
from preprocessing_module import preprocess_image
import mediapipe as mp

import cv2
import mediapipe as mp
import numpy as np
import pickle
import threading

app = Flask(__name__)

model_path = r'C:\Users\Yashaswini\OneDrive\Desktop\Project\Indian Sign Language Web Application\lightgbm_model.pkl'
gbm_model = joblib.load(model_path)

model_dict = pickle.load(open('./model_dummy.p', 'rb'))
model = model_dict['model_dummy']

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

labels_dict = {0: '1', 1: '2', 2: '3', 3: '4', 4: '5', 5: '6', 6: '7', 7: '8', 8: '9',
               9:'A',10:'B',11:'C',12:'D',13:'E',14:'F',15:'G',16:'H',17:'I',18:'J',19:'K',20:'L',
               21:'M',22:'N',23:'O',24:'P',25:'Q',26:'R',27:'S',28:'T',29:'U',30:'V',31:'W',
               32:'X',33:'Y',34:'Z'}

# Initialize variables for threading
lock = threading.Lock()
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Set the width of the frame
cap.set(4, 480)  # Set the height of the frame
skip_frames = 3  # Skip processing for every 5 frames
frame_count = 0
capture_image = False

def capture_frame():
    global frame_count
    while True:
        ret, current_frame = cap.read()
        if not ret:
            break

        # Skip frames
        frame_count += 1
        if frame_count % skip_frames != 0:
            continue

        frame_rgb = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    current_frame,  # image to draw
                    hand_landmarks,  # model output
                    mp_hands.HAND_CONNECTIONS,  # hand connections
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            data_aux = []
            x_ = []
            y_ = []

            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            x1 = int(min(x_) * current_frame.shape[1]) - 10
            y1 = int(min(y_) * current_frame.shape[0]) - 10

            x2 = int(max(x_) * current_frame.shape[1]) - 10
            y2 = int(max(y_) * current_frame.shape[0]) - 10

            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]

            cv2.rectangle(current_frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(current_frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                        cv2.LINE_AA)

            _, jpeg = cv2.imencode('.jpg', current_frame)
            frame_bytes = jpeg.tobytes()

            with lock:
                frame = current_frame.copy()
                capture_image = False

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/capture_image', methods=['POST'])
def capture_image():
    global capture_image
    capture_image = True
    return 'Image Captured'

@app.route('/stop_capture', methods=['POST'])
def stop_capture():
    global cap
    cap.release()
    return 'Capture Stopped'

@app.route('/video_feed')
def video_feed():
    return Response(capture_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('error.html', message='No file part')

        file = request.files['image']

        if file.filename == '':
            return render_template('error.html', message='No selected file')

        # Save the file to a location and get the file path
        file_path = r'C:\Users\Yashaswini\OneDrive\Desktop\Project\unused_Files' + secure_filename(file.filename)
        file.save(file_path)

        # Call the preprocess function and get the preprocessed file path
        preprocessed_path = preprocess_image(file_path)

        # Process the image and extract features
        features = extract_features(preprocessed_path)

        # Convert features to a DataFrame
        features_df = pd.DataFrame(features.reshape(1, -1))

        # Use the pre-trained Random Forest model for prediction
        prediction = gbm_model.predict(features_df)

        # Map the predicted label using labels_dict
        predicted_label = labels_dict[prediction[0]]

        # Render the prediction in the HTML template
        return render_template('result.html', prediction=predicted_label)

    return render_template('upload.html')

@app.route('/realtime')
def realtime():
    return render_template('realtime.html')

if __name__ == '__main__':
    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
    app.run(debug=True)
