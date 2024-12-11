import cv2
import mediapipe as mp
from mediapipe_utils.detection import mediapipe_detection, extract_keypoints, save_keypoints
from mediapipe_utils.drawing import draw_styled_landmarks

os.makedirs('keypoints', exist_ok=True)
mp_holistic = mp.solutions.holistic

cap = cv2.VideoCapture(0)

# Set mediapipe model for holistic (including face, hands, and pose)
with mp_holistic.Holistic(
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5
) as holistic:
    frame_count = 0
    while cap.isOpened():
        # Read feed
        ret, frame = cap.read()

        # Make detections
        image, results = mediapipe_detection(frame, holistic)

        # Draw styled landmarks
        draw_styled_landmarks(image, results)

        # Show to screen
        cv2.imshow("OpenCV Feed", image)

        # Extract and save keypoints if landmarks are detected
        if results.pose_landmarks or results.face_landmarks or results.left_hand_landmarks or results.right_hand_landmarks:
            # Save keypoints with frame number
            save_keypoints(results, f'keypoints/{frame_count}')
            frame_count += 1

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
