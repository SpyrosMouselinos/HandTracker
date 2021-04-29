import math
from ctypes import cast, POINTER

import cv2
import numpy as np
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

from Modules.HandTracker import HandTracker as HT

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

detector = HT(maxhands=1, detectioncon=0.8)


class AudioController:
    def __init__(self, com_code=7):
        self.com_code = com_code
        self.devices = AudioUtilities.GetSpeakers()
        self.interface = self.devices.Activate(IAudioEndpointVolume._iid_, 7, None)
        self.volume = cast(self.interface, POINTER(IAudioEndpointVolume))
        self.minVol, self.maxVol = self.volume.GetVolumeRange()[0:2]
        self.input_range = [self.minVol, self.maxVol]

    def set_volume(self, volume):
        scaled_volume = np.interp(volume, self.input_range, [self.minVol, self.maxVol])
        self.volume.SetMasterVolumeLevel(scaled_volume, None)
        return

    def set_range(self, custom_range):
        """
        Registers a custom sensor range and maps it to volume range
        so that set_volume can be called on the custom range
        """
        self.input_range = [custom_range[0], custom_range[1]]
        return

    def reset_range(self):
        self.input_range = [self.minVol, self.maxVol]
        return


def test_audio_controller():
    controller = AudioController()
    controller.set_range([50, 300])
    volBar = 400
    volPer = 0
    while True:
        success, img = cap.read()
        img = detector.findhands(img)
        lmList = detector.findposition(img, draw=False)
        if len(lmList) != 0:

            x1, y1 = lmList[4][1], lmList[4][2]  # 4 = Thumb Tip
            x2, y2 = lmList[8][1], lmList[8][2]  # 8 = Index Finger Tip
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            cv2.circle(img, (x1, y1), 5, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 5, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            length = math.hypot(x2 - x1, y2 - y1)

            volBar = np.interp(length, [50, 300], [400, 150])
            volPer = np.interp(length, [50, 300], [0, 100])
            controller.set_volume(length)

            if length < 50:
                cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)

        cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
        cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0), cv2.FILLED)
        cv2.putText(img, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX,
                    1, (255, 0, 0), 3)

        cv2.imshow("Img", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    test_audio_controller()
