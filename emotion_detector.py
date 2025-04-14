import cv2
import numpy as np
import tensorflow as tf
import os
import csv
from datetime import datetime
import pandas as pd

class EmotionDetector:
    def __init__(self, model=None):
        # Store the provided model or load it if not provided
        if model is not None:
            self.model = model
        else:
            # Load the model architecture and weights
            model_json_path = os.path.join(os.path.dirname(__file__), "model.json")
            model_weights_path = os.path.join(os.path.dirname(__file__), "model.weights.h5")
            
            # Load the model architecture
            with open(model_json_path, "r") as json_file:
                model_json = json_file.read()
                self.model = tf.keras.models.model_from_json(model_json)
            
            # Load the weights
            self.model.load_weights(model_weights_path)
        
        # Define emotion labels
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        
        # Load face detection cascade
        cascade_path = os.path.join(os.path.dirname(__file__), "haarcascade_frontalface_default.xml")
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Define input shape for the model
        self.input_shape = (48, 48, 1)
        
        # Initialize CSV file
        self.csv_file = 'emotion_log.csv'
        self.initialize_csv()
        
        # Define stress levels for emotions
        self.stress_levels = {
            'Angry': 5,
            'Disgust': 4,
            'Fear': 4,
            'Happy': 1,
            'Sad': 3,
            'Surprise': 2,
            'Neutral': 2
        }
        
    def initialize_csv(self):
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Timestamp', 'Emotion', 'Confidence', 'Stress Level', 'Notes'])
    
    def log_emotion(self, emotion, confidence):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        stress_level = self.stress_levels.get(emotion, 2)  # Default to 2 if emotion not found
        
        # Add notes based on stress level
        notes = self.get_stress_notes(stress_level)
        
        with open(self.csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, emotion, confidence, stress_level, notes])
    
    def get_stress_notes(self, stress_level):
        notes = {
            1: "Low stress - Good emotional state",
            2: "Moderate stress - Normal range",
            3: "Elevated stress - Consider taking a break",
            4: "High stress - Recommended to practice stress management",
            5: "Very high stress - Consider seeking support"
        }
        return notes.get(stress_level, "Unknown stress level")
    
    def analyze_stress_trend(self):
        try:
            df = pd.read_csv(self.csv_file)
            if len(df) > 0:
                avg_stress = df['Stress Level'].mean()
                recent_stress = df['Stress Level'].tail(5).mean()
                print("\nStress Analysis:")
                print(f"Average Stress Level: {avg_stress:.2f}")
                print(f"Recent Stress Level (last 5): {recent_stress:.2f}")
                
                if recent_stress > avg_stress:
                    print("⚠️ Warning: Your stress levels have been increasing recently.")
                elif recent_stress < avg_stress:
                    print("✅ Good news: Your stress levels have been decreasing recently.")
        except Exception as e:
            print(f"Error analyzing stress trend: {str(e)}")
    
    def preprocess_image(self, image_data):
        # Convert image data to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            return None
            
        # Get the first face
        x, y, w, h = faces[0]
        face = gray[y:y+h, x:x+w]
        
        # Resize to 48x48
        face = cv2.resize(face, (48, 48))
        
        # Normalize
        face = face / 255.0
        
        # Reshape for model input
        face = np.expand_dims(face, axis=0)
        face = np.expand_dims(face, axis=-1)
        
        return face
        
    def detect_emotion(self, image_data):
        # Preprocess the image
        processed_image = self.preprocess_image(image_data)
        
        if processed_image is None:
            return None, 0.0
            
        # Make prediction
        prediction = self.model.predict(processed_image)
        
        # Get emotion and confidence
        emotion_idx = np.argmax(prediction[0])
        confidence = prediction[0][emotion_idx]
        
        # Log the emotion
        self.log_emotion(self.emotion_labels[emotion_idx], confidence)
        
        return self.emotion_labels[emotion_idx], confidence