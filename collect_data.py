import cv2
import mediapipe as mp
import csv

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

LABELS = {
    "h": "hello",
    "y": "yes",
    "n": "no",
    "t": "thanks",
    "l": "i love you",
    "m": "mom",
    "d": "dad"
    } #add more here for more gestures

cap = cv2.VideoCapture(0)
csv_file = open("sign_data.csv", mode="w", newline="")
csv_writer = csv.writer(csv_file)

# Header: label + x,y,z for 21 landmarks = 63 features
csv_writer.writerow(["label"] + [f"{axis}{i}" for i in range(21) for axis in ("x","y","z")])

print("Press h/y/n while showing the sign. Press 'q' to quit.")

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            key = cv2.waitKey(1) & 0xFF
            if key in [ord(k) for k in LABELS]:
                label = LABELS[chr(key)]
                print(f"Captured: {label}")
                csv_writer.writerow([label] + landmarks)
    else:
        key = cv2.waitKey(1) & 0xFF

    cv2.imshow("Collecting Data", frame)

    if key == ord("q"):
        break

cap.release()
csv_file.close()
cv2.destroyAllWindows()
