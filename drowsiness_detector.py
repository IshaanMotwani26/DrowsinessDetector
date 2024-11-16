import cv2
import pygame
import time
from pygame import mixer
import sys
import os
import numpy as np
import wave

def initialize_sound():
    try:
        mixer.init()
        # Check if alert.wav exists, if not create a simple beep using numpy
        if not os.path.exists('alert.wav'):
            # Generate a simple beep sound
            sample_rate = 44100
            duration = 1.0
            frequency = 2500
            t = np.linspace(0, duration, int(sample_rate * duration))
            samples = np.sin(2 * np.pi * frequency * t)
            samples = (samples * 32767).astype(np.int16)
            
            # Create and save a temporary wav file
            with wave.open('temp_alert.wav', 'w') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(samples.tobytes())
            
            mixer.music.load('temp_alert.wav')
        else:
            mixer.music.load('alert.wav')
    except Exception as e:
        print(f"Error initializing sound: {e}")
        sys.exit(1)

def play_alarm():
    try:
        mixer.music.play()
    except:
        print("Could not play alert sound")

def initialize_camera():
    try:
        # First try to find the built-in webcam
        for index in [-1, 0, 1]:  # Try -1 for default, 0, and 1
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                # Get camera properties
                backend = cap.getBackendName()
                width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                
                print(f"Testing camera {index}:")
                print(f"Backend: {backend}")
                print(f"Resolution: {width}x{height}")
                
                # Test if we can actually read from this camera
                ret, frame = cap.read()
                if ret:
                    # If it's the built-in camera (typically 1280x720 or similar)
                    if width >= 640 and "Continuity" not in backend:
                        print(f"Successfully connected to built-in camera (index {index})")
                        # Set properties
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                        return cap
                cap.release()
        
        # If we get here, try one last time with explicit properties
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            # Try to force QTKit/AVFoundation backend on macOS
            if sys.platform == 'darwin':
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
            
            ret, frame = cap.read()
            if ret:
                print("Using default camera with forced settings")
                return cap
                
        raise Exception("Could not initialize any suitable camera")
        
    except Exception as e:
        print(f"Error accessing camera: {e}")
        print("Please make sure camera permissions are granted")
        print("Try disconnecting any external cameras or iPhone")
        sys.exit(1)

def load_cascades():
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        if face_cascade.empty() or eye_cascade.empty():
            raise Exception("Error loading cascade classifiers")
            
        return face_cascade, eye_cascade
    except Exception as e:
        print(f"Error loading cascade files: {e}")
        sys.exit(1)

def main():
    # Initialize camera with error handling
    cap = initialize_camera()
    
    # Load cascade classifiers
    face_cascade, eye_cascade = load_cascades()
    
    # Initialize sound
    initialize_sound()
    
    # Variables for tracking eye state
    eyes_closed_time = 0
    eyes_closed_threshold = 2.0  # Seconds before alert
    
    print("Drowsiness detection started. Press 'q' to quit.")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.3,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            eyes_detected = False
            
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                
                # Detect eyes in the face region
                eyes = eye_cascade.detectMultiScale(
                    roi_gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(20, 20)
                )
                
                if len(eyes) >= 2:  # If at least two eyes are detected
                    eyes_detected = True
                    eyes_closed_time = 0  # Reset the counter
                    
                    # Draw rectangles around eyes
                    for (ex, ey, ew, eh) in eyes:
                        cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            if not eyes_detected:
                if eyes_closed_time == 0:
                    eyes_closed_time = time.time()
                elif time.time() - eyes_closed_time >= eyes_closed_threshold:
                    play_alarm()
                    cv2.putText(frame, "Wake Up!", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imshow('Drowsiness Detection', frame)
            
            # Break loop with 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Quitting application...")
                break
                
    except Exception as e:
        print(f"An error occurred: {e}")
    
    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 