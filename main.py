import os
import cv2
import mediapipe as mp
import numpy as np
from mediapipe_utils.detection import (
    mediapipe_detection,
    extract_keypoints,
    save_keypoints,
)
from folder_setup import setupFolder
from mediapipe_utils.drawing import draw_styled_landmarks

DATA_PATH = os.path.join("MP_Data")
actions = np.array(["hello", "thanks", "iloveyou"])
start_folder = 0
no_sequences = 30
sequence_length = 30
mp_holistic = mp.solutions.holistic

cap = cv2.VideoCapture(0)


# Set mediapipe model for holistic (including face, hands, and pose)

stop_capture = False  # Flag to stop capturing

with mp_holistic.Holistic(
    min_detection_confidence=0.5, min_tracking_confidence=0.5
) as holistic:
    for action in actions:  # Loop through each action
        if stop_capture:
            break
        for sequence in range(
            start_folder, start_folder + no_sequences
        ):  # Loop through sequences
            if stop_capture:
                break
            for frame_num in range(sequence_length):  # Loop through frames
                # Read feed
                ret, frame = cap.read()

                # Make detections
                image, results = mediapipe_detection(frame, holistic)

                # Draw landmarks
                draw_styled_landmarks(image, results)

                # Apply wait logic for the first frame
                if frame_num == 0:
                    cv2.putText(
                        image,
                        "STARTING COLLECTION",
                        (120, 200),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        4,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        image,
                        f"Collecting frames for {action} Video Number {sequence}",
                        (15, 12),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        1,
                        cv2.LINE_AA,
                    )
                    cv2.imshow("OpenCV Feed", image)
                    cv2.waitKey(500)  # Pause for 500ms
                else:
                    cv2.putText(
                        image,
                        f"Collecting frames for {action} Video Number {sequence}",
                        (15, 12),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        1,
                        cv2.LINE_AA,
                    )
                    cv2.imshow("OpenCV Feed", image)

                # Export keypoints
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(
                    DATA_PATH, action, str(sequence), str(frame_num)
                )
                save_keypoints(keypoints, npy_path)

                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord("q"):
                    stop_capture = True
                    break

cap.release()  # Release webcam
cv2.destroyAllWindows()  # Close all OpenCV windows
