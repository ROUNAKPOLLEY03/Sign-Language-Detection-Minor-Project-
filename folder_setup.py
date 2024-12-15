import os
import numpy as np

# Define constants
DATA_PATH = os.path.join("MP_Data")
actions = np.array(["hello", "thanks", "iloveyou"])
no_sequences = 30  # Number of videos per action


def setupFolder():
    for action in actions:
        # Ensure the action folder exists
        action_path = os.path.join(DATA_PATH, action)
        if not os.path.exists(action_path):
            os.makedirs(action_path)

        # Find the maximum folder number or default to 0
        dirmax = (
            np.max(np.array(os.listdir(action_path)).astype(int))
            if os.listdir(action_path)
            else 0
        )

        # Create new sequence folders
        for sequence in range(1, no_sequences + 1):
            try:
                os.makedirs(os.path.join(action_path, str(dirmax + sequence)))
            except FileExistsError:
                pass
