import cv2
import mediapipe as mp
from mediapipe_utils.detection import mediapipe_detection
from mediapipe_utils.drawing import draw_styled_landmarks


mp_holistic = mp.solutions.holistic

cap = cv2.VideoCapture(0)

# Set mediapipe model for holistic (including face, hands, and pose)
with mp_holistic.Holistic(
    min_detection_confidence=0.5, min_tracking_confidence=0.5
) as holistic:
    while cap.isOpened():
        # Read feed
        ret, frame = cap.read()

        # Make detections
        image, results = mediapipe_detection(frame, holistic)

        # Draw styled landmarks
        draw_styled_landmarks(image, results)

        # Show to screen
        cv2.imshow("OpenCV Feed", image)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
