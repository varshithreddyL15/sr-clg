import cv2

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

# Capture video from the webcam55
cap = cv2.VideoCapture(0)

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
            score = dominant_emotion[emotion_name] * 100  # Convert to percentage

            # Prepare the text overlay
            overlay = frame.copy()

            # Define text box position and size
            text_x = 10
            text_y = 30
            box_width = 300  # Width of the box
            box_height = 70  # Height of the box for a single line

            # Draw a filled rectangle for background
            cv2.rectangle(overlay, (text_x - 5, text_y - 5), (text_x + box_width, text_y + box_height), (0, 0, 0), -1)

            # Display the dominant emotion and score
            cv2.putText(overlay, f'Emotion: {emotion_name}', (text_x, text_y + 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(overlay, f'Score: {score:.2f}%', (text_x, text_y + 45), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

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
