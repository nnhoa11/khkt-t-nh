import threading
import numpy as np
import tensorflow as tf
import cv2
import mediapipe as mp
import keras
# from tensorflow import
# import keras
# from HandDetectorModule import HandDetector
# from serial import Serial
import imutils


# serialcomm = Serial('COM3', 9600)
# serialcomm.timeout = 1
current = ""

class HandDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, minTrackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.minTrackCon = minTrackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode, max_num_hands=self.maxHands,
                                        min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence=self.minTrackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
        self.fingers = []
        self.lmList = []

    def findHands(self, img, draw=True, flipType=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        allHands = []
        h, w, c = img.shape
        if self.results.multi_hand_landmarks:
            for handType, handLms in zip(self.results.multi_handedness, self.results.multi_hand_landmarks):
                myHand = {}
                ## lmList
                mylmList = []
                xList = []
                yList = []
                for id, lm in enumerate(handLms.landmark):
                    px, py, pz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                    mylmList.append([px, py])
                    xList.append(px)
                    yList.append(py)

                ## bbox
                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                boxW, boxH = xmax - xmin, ymax - ymin
                bbox = xmin, ymin, boxW, boxH
                cx, cy = bbox[0] + (bbox[2] // 2), \
                         bbox[1] + (bbox[3] // 2)

                myHand["lmList"] = mylmList
                myHand["bbox"] = bbox
                myHand["center"] = (cx, cy)

                if flipType:
                    if handType.classification[0].label == "Right":
                        myHand["type"] = "Left"
                    else:
                        myHand["type"] = "Right"
                else:
                    myHand["type"] = handType.classification[0].label
                allHands.append(myHand)

                ## draw
                if draw:
                    global current
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
                    cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20),
                                  (bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20),
                                  (255, 0, 255), 2)
                    cv2.putText(img, current, (bbox[0] - 30, bbox[1] - 30), cv2.FONT_HERSHEY_PLAIN,
                                2, (255, 0, 255), 2)
        if draw:
            return allHands, img
        else:
            return allHands



model = tf.keras.models.load_model('hackathon_model_1.h5')
label = ['I love you', 'I', 'Give me my money', 'I want']

label_map = [['I love you', [1,1,1,1,1,1]],
             ['I', [0,0,0,0,0,1]],
             ['Give me my money', [1,1,1,1,1,0]],
             ['I want', [0,0,0,0,0,0]]]
def get_lmList(hand):
    c_lmlist = []
    # print(hand['lmList'])
    for count, lm in enumerate(hand['lmList']):
        # print(lm)
        c_lmlist.append(lm[0])
        c_lmlist.append(lm[1])

    # print(type(c_lmlist))
    return c_lmlist

def detect(model, lm_list, type):
    global label
    # print(len(lm_list))
    # lm_list = np.array(lm_list)
    lm_list_tensor = np.expand_dims(lm_list, axis=0)

    # print(lm_list_tensor.shape)
    results = model.predict(lm_list_tensor)
    # if results > 0.5 : print("pose1")
    # else : print("pose2")
    result = np.argmax(results)
    # if (type != "Left"):
    #     if (label_map[result][1][5] == 0):
    #         result = result - 2
    #     else: result = result + 2

    fingers1 = label_map[result][1]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Use putText() method for
    # inserting text on video
    global current

    if current != label_map[result][0]:
        current = label_map[result][0]
    # serialcomm.write(s.encode())

cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon = 0.7, maxHands = 2)
no_time_step = 2
number_of_step = 300
counts  = 0
lmList = []
my_list = []
while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=1000, height=1000)
    # cv2.rectangle(frame, (0, 0), (250, 100), (0, 0, 0), -1)
    hand, frame = detector.findHands(frame)
    cv2.imshow('cam', frame)
    if counts > 6 * no_time_step:
        # print('start detecting...')
        if hand:

            my_list = get_lmList(hand[0])
            lmList.append(my_list)
            if (len(lmList) == no_time_step):
                t1 = threading.Thread(target=detect, args=(model, lmList, hand[0]["type"]))
                t1.start()
                lmList = []
    counts += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# model.summary()
cap.release()
cv2.destroyAllWindows()
