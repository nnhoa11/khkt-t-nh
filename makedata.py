import pandas as pd
import cv2
from HandDetectorModule import HandDetector
label = [
"000011",
]
def get_lmList(hand):
    c_lmlist = []
    # print(hand['lmList'])
    for count, lm in enumerate(hand['lmList']):
        # print(hand['lmList'])
        c_lmlist.append(lm[0])
        c_lmlist.append(lm[1])
        # c_lmlist.append(lm[2])
    # print(type(c_lmlist))
    return c_lmlist



for i in range(0,4):
    cap = cv2.VideoCapture(0)
    detector = HandDetector(detectionCon=0.7, maxHands=1)
    no_time_step = 2
    number_of_step = 200
    counts = 0
    lmList = []
    my_list = []
    breakSwitch = False
    while counts <= number_of_step:
        ret, frame = cap.read()
        hand, frame= detector.findHands(frame)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.rectangle(frame, (0, 0), (1000, 50), (255, 255, 255), -1)
        cv2.putText(frame,
                    str(i),
                    (50, 50),
                    font, 1,
                    (0, 0, 0),
                    2,
                    cv2.LINE_4)
        cv2.imshow('cam', frame)
        if hand:
            # for step in range(0, number_of_step):
            for count in range(0, no_time_step):
                # lmList.append(get_lmList(hand[0]))
                lmArr = get_lmList(hand[0])
                lmList.append(lmArr)
            # print(len(lmList))
            # if len()
            print(counts)
            counts += 1
            my_list.append(lmList)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            breakSwitch = True
            break

    if breakSwitch: break

    df = pd.DataFrame(lmList)
    df.to_csv("./pose4-right.txt")
    data = pd.read_csv("./pose4-right.txt")
    print(data.iloc[:0:].values)
    cap.release()
    cv2.destroyAllWindows()
