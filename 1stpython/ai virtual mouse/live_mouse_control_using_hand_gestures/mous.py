import cv2
import pyttsx3
import time

# Attempt to import the FER library and handle the case where it's not installed
try:
    from fer import FER
except ModuleNotFoundError:
    print("The 'fer' module is not installed. Please install it using 'pip install fer'.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred while importing FER: {e}")
    exit()

# Initialize the emotion detector
detector = FER()

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Capture video from the webcam
cap = cv2.VideoCapture(0)

# Set lower resolution (e.g., 640x480) for better performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Store the last spoken emotion to avoid repetition
last_spoken_emotion = None
last_speak_time = time.time()  # Track the last time the emotion was spoken

# Define text box position and size
text_x = 10
text_y = 30
box_width = 300  # Width of the box
box_height = 70  # Height of the box for a single line

# Frame counter to skip detection on some frames for performance boost
frame_counter = 0
frames_to_skip = 5  # Adjust this number for your system; higher skips more frames

try:
    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture video. Exiting...")
            break

        # Increase frame counter and check if we should process this frame
        frame_counter += 1
        if frame_counter % frames_to_skip == 0:
            # Detect emotions in the current frame
            emotions = detector.detect_emotions(frame)
            overlay = frame.copy()  # Copy frame to create overlay

            # Draw a filled rectangle for background in pink
            cv2.rectangle(overlay, (text_x - 5, text_y - 5), 
                          (text_x + box_width, text_y + box_height), (255, 105, 180), -1)

            if emotions:
                # Get the dominant emotion for the first detected face
                dominant_emotion = emotions[0]['emotions']
                emotion_name = max(dominant_emotion, key=dominant_emotion.get)
                emotion_score = dominant_emotion[emotion_name]

                # Display the dominant emotion and its score
                cv2.putText(overlay, f'Emotion: {emotion_name} ({emotion_score:.2f})', 
                            (text_x, text_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 
                            (255, 255, 255), 2, cv2.LINE_AA)

                # Speak the emotion if it has changed and after a delay
                if emotion_name != last_spoken_emotion and (time.time() - last_speak_time > 2):
                    engine.say(f"Emotion: {emotion_name}")
                    engine.runAndWait()
                    last_spoken_emotion = emotion_name
                    last_speak_time = time.time()
            else:
                # If no emotions detected, display "No Emotion"
                cv2.putText(overlay, 'Emotion: No Emotion', 
                            (text_x, text_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 
                            (255, 255, 255), 2, cv2.LINE_AA)

            # Display the overlay on the frame
            cv2.imshow("Emotion Detection", overlay)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
