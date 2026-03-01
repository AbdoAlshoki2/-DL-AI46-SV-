import cv2
import joblib
import mediapipe as mp
import numpy as np
import torch
import yaml
from pathlib import Path

from gesture_net import GestureNet
from transformers import HandCentering, HandNormalization


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_model(checkpoint_path: str) -> tuple:
    """Load GestureNet from a .pth checkpoint. Returns (model, class_names)."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model = GestureNet(
        input_size=checkpoint['input_size'],
        num_classes=checkpoint['num_classes'],
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint['class_names']


def main():
    config = load_config()

    print(f"Loading feature processor from: {config['processors']['feature_processor']}")
    feature_processor = joblib.load(config['processors']['feature_processor'])

    print(f"Loading model checkpoint from: {config['model_path']}")
    model, class_names = load_model(config['model_path'])
    print(f"Classes: {class_names}")

    mp_hands = mp.solutions.hands
    hands    = mp_hands.Hands()

    display = config['display']
    font    = getattr(cv2, display['font'])

    cap = cv2.VideoCapture(0)
    print("Starting hand detection... Press 'q' to quit")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        landmarks_buf = np.zeros((1, 63), dtype=np.float32)
        rgb_frame     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results       = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

                for i, lm in enumerate(hand_landmarks.landmark):
                    landmarks_buf[0, i * 3]     = lm.x
                    landmarks_buf[0, i * 3 + 1] = lm.y
                    landmarks_buf[0, i * 3 + 2] = lm.z

                # Preprocess and infer
                processed = feature_processor.transform(landmarks_buf).astype(np.float32)
                with torch.no_grad():
                    logits    = model(torch.from_numpy(processed))
                    pred_idx  = logits.argmax(dim=1).item()
                    label_str = class_names[pred_idx]

                y_offset   = display['position'][1] + (hand_idx * 40)
                label_text = f"Hand {hand_idx + 1}: {label_str}"

                cv2.putText(
                    frame,
                    label_text,
                    (display['position'][0], y_offset),
                    font,
                    display['font_scale'],
                    tuple(display['color']),
                    display['thickness'],
                )

        cv2.imshow("Hand Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()


if __name__ == "__main__":
    main()
