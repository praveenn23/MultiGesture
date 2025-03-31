import cv2
import mediapipe as mp
import math
import numpy as np
import pyautogui
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Mediapipe setup
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Volume Control Setup
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
minVol, maxVol, _ = volume.GetVolumeRange()

# Screen and Camera Setup
pyautogui.FAILSAFE = False
wScrn, hScrn = pyautogui.size()
wCam, hCam = 640, 480
smoothening = 5
frameR = 100  # Frame boundary

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

plocX, plocY = 0, 0
clocX, clocY = 0, 0

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=1) as hands:
    while cap.isOpened():
        success, img = cap.read()
        if not success:
            continue

        img = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB)
        img.flags.writeable = False
        results = hands.process(img)
        img.flags.writeable = True
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        lmList = []
        handType = None

        if results.multi_hand_landmarks:
            for hand in results.multi_handedness:
                handType = hand.classification[0].label
            
            for hand_landmarks in results.multi_hand_landmarks:
                for id, lm in enumerate(hand_landmarks.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        if len(lmList) != 0:
            x1, y1 = lmList[4][1], lmList[4][2]  # Thumb tip
            x2, y2 = lmList[8][1], lmList[8][2]  # Index finger tip
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            if handType == 'Left':  # Volume Control
                d = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                vol = np.interp(d, [30, 270], [minVol, maxVol])  # Adjusted for better control
                volume.SetMasterVolumeLevel(vol, None)
                volBar = np.interp(d, [30, 270], [400, 150])
                volPer = np.interp(d, [30, 270], [0, 100])  # Fixed 0-100% range
                
                # Draw volume bar
                cv2.rectangle(img, (50, 150), (85, 400), (0, 0, 0), 3)
                cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0), cv2.FILLED)
                cv2.putText(img, f'{int(volPer)}%', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 3)

                # Highlight fingers used for volume control
                cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)  # Red for thumb
                cv2.circle(img, (x2, y2), 10, (0, 255, 0), cv2.FILLED)  # Green for index finger
                cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)  # Blue line for reference

            elif handType == 'Right':  # Mouse Control
                x3 = np.interp(x2, (frameR, wCam - frameR), (0, wScrn))
                y3 = np.interp(y2, (frameR, hCam - frameR), (0, hScrn))
                clocX = plocX + (x3 - plocX) / smoothening
                clocY = plocY + (y3 - plocY) / smoothening
                pyautogui.moveTo(clocX, clocY, duration=0)
                plocX, plocY = clocX, clocY
                
                # Click when index finger and middle finger are close
                distance = math.dist([lmList[8][1], lmList[8][2]], [lmList[12][1], lmList[12][2]])
                if distance < 30:
                    pyautogui.click()
                    cv2.circle(img, (x2, y2), 10, (0, 255, 0), cv2.FILLED)

        cv2.imshow('Hand Gesture Control', img)
        if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
            break

cap.release()
cv2.destroyAllWindows()
