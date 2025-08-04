import cv2
import pyttsx3

# Attempt to import the FER library and handle the case where it's not installed
try:
    from fer import FER
except ModuleNotFoundError:
    print("The 'fer' module is not installed. Please install it using 'pip install fer'.")
    exit()  # Exit the program if FER is not available
except Exception as e:
    print(f"An unexpected error occurred while importing FER: {e}")
    exit()  # Exit if there's another issue

# Initialize the emotion detector
detector = FER()

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Capture video from the webcam
cap = cv2.VideoCapture(0)

# Store the last spoken emotion to avoid repetition
last_spoken_emotion = None

try:
    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture video. Exiting...")
            break

        # Detect emotions in the frame
        emotions = detector.detect_emotions(frame)

        if emotions:
            # Get the dominant emotion for the first detected face
            dominant_emotion = emotions[0]['emotions']
            # Find the emotion with the highest score
            emotion_name = max(dominant_emotion, key=dominant_emotion.get)

            # Prepare the text overlay
            overlay = frame.copy()

            # Define text box position and size
            text_x = 10
            text_y = 30
            box_width = 300  # Width of the box
            box_height = 70  # Height of the box for a single line

            # Draw a filled rectangle for background in pink
            cv2.rectangle(overlay, (text_x - 5, text_y - 5), (text_x + box_width, text_y + box_height), (255, 105, 180), -1)  # Pink color in BGR

            # Display the dominant emotion in white
            cv2.putText(overlay, f'Emotion: {emotion_name}', (text_x, text_y + 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)  # White text

            # Speak the emotion in the desired format if it has changed
            if emotion_name != last_spoken_emotion:
                engine.say(f"Emotion: {emotion_name}")  # Pronounce "Emotion: [emotion name]"
                engine.runAndWait()
                last_spoken_emotion = emotion_name

        else:
            # Clear overlay to remove any previous text if no emotions are detected
            overlay = frame.copy()

        # Display the overlay on the frame
        cv2.imshow("Emotion Detection", overlay)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
