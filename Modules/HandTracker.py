import cv2
import mediapipe as mp


class HandTracker:
    def __init__(self, mode=False, maxhands=2, detectioncon=0.5, trackcon=0.5):
        self.mode = mode
        self.maxhands = maxhands
        self.detectioncon = detectioncon
        self.trackcon = trackcon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxhands,
                                        self.detectioncon, self.trackcon)
        self.mpDraw = mp.solutions.drawing_utils
        self.results = None

    def findhands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
        return img

    def findposition(self, img, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            for id, lm in enumerate(self.results.multi_hand_landmarks[0].landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

        return lmList

    def count_fingers(self):
        return


def test_tracker(verbose=0):
    cap = cv2.VideoCapture(0)
    detector = HandTracker()
    while True:
        success, img = cap.read()
        img = detector.findhands(img)
        lmList = detector.findposition(img)
        if verbose:
            for xx in lmList:
                print(f"Item: {xx[0]}, Pos X: {xx[1]}, Pos Y: {xx[2]}")

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def test_counter():
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    detector = HandTracker(maxhands=1, detectioncon=0.8)

    tipIds = [4, 8, 12, 16, 20]
    threshIds = [15, 15, 15, 15, 15]

    while True:
        success, img = cap.read()
        img = detector.findhands(img)
        lmList = detector.findposition(img, draw=False)

        if len(lmList) != 0:
            print(lmList[4], lmList[2])
            fingers = 0

            # Thumb
            if abs(lmList[4][1] - lmList[2][1]) >= threshIds[0]:
                fingers += 1
            # Rest Fingers
            for id in range(1, 5):
                if abs(lmList[tipIds[id]][2] - lmList[tipIds[id] - 2][2]) >= threshIds[id]:
                    fingers += 1

            cv2.putText(img, str(fingers), (45, 375), cv2.FONT_ITALIC,
                        10, (255, 0, 0), 25)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    test_counter()
