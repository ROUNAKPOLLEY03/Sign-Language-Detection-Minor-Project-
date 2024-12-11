import cv2
import numpy as np

def mediapipe_detection(image, model):
    """
        Convert image color space and process with MediaPipe model.
        
        Args:
            image: Input image in BGR color space
            model: MediaPipe holistic model
        
        Returns:
            Tuple of processed image and detection results
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False  # Image is no longer writeable
    results = model.process(image)  # Make prediction
    image.flags.writeable = True  # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR COVERSION RGB 2 BGR
    return image, results

def extract_keypoints(results):
    """
        Extract keypoints from MediaPipe results for pose, face, and hands.
        
        Args:
            results: MediaPipe detection results
        
        Returns:
            numpy array of concatenated keypoints
    """
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() \
        if results.pose_landmarks else np.zeros(33*4)
    
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() \
        if results.face_landmarks else np.zeros(468*3)
    
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() \
        if results.left_hand_landmarks else np.zeros(21*3)
    
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() \
        if results.right_hand_landmarks else np.zeros(21*3)
    
    return np.concatenate([pose, face, lh, rh])

def save_keypoints(results, filename='keypoints'):
    """
        Save extracted keypoints to a .npy file.
        
        Args:
            results: MediaPipe detection results
            filename: Name of the file to save (without .npy extension)
    """
    keypoints = extract_keypoints(results)
    np.save(filename, keypoints)
    return keypoints
