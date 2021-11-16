import threading
import time
import cv2
from utils.modules import *


capturedFrame=[]
after_face_detection=[]
detected_faces=[]
after_mobile_detection=[]
after_spoof_detection=[]
after_gaze_detection=[]
msg=[]
finalResult=[]
is_process_completed = False

configPath = 'model/yolov3.cfg'
weightsPath = 'model/yolov3.weights'
net = cv2.dnn.readNet(weightsPath, configPath)

# capturing the image from webcam
def task0():
    global is_process_completed
    vid = cv2.VideoCapture(0)
    while True:
        ret, image = vid.read()
        if not ret:
            is_process_completed = True
            break
        after_mobile_detection.append(image)
        time.sleep(1)
        if is_process_completed:
            break
    vid.release()


def task1():
    global is_process_completed
    while True:
        if capturedFrame:
            image = capturedFrame[0]
            capturedFrame.pop(0)
            found = mobile_detection(image,net)
            if found:
                print("mobile found!!")
                msg.append("mobile found!!")
                finalResult.append(image)
                continue
            else:
                print("good")
            after_mobile_detection.append(image)
        if is_process_completed:
            break

#
def task2():
    global is_process_completed
    while True:
        if len(after_mobile_detection):
            image = after_mobile_detection[0]
            after_mobile_detection.pop(0)
            no_faces, faces= face_detection(image)
            if no_faces == 0:
                print("Face not found!!")
                msg.append("Face not found!!")
                finalResult.append(image)
                continue
            elif no_faces > 1:
                print("More than one person found!!")
                msg.append("More than one person found!!")
                finalResult.append(image)
                continue
            after_face_detection.append(image)
            detected_faces.append(faces)
        if is_process_completed:
            break

def task3():
    global is_process_completed
    while True:
        if len(after_face_detection):
            image = after_face_detection[0]
            after_face_detection.pop(0)
            face=detected_faces.pop(0)
            is_spoof= spoof_detection(face[0],image)
            if is_spoof:
                print("printed spoof image")
                msg.append("printed spoof image")
                finalResult.append(image)
                continue
            print("not spoof image")
            after_spoof_detection.append(image)
        if is_process_completed:
            break
def task4():
    global is_process_completed
    while True:
        if len(after_spoof_detection):
            image = after_spoof_detection[0]
            after_spoof_detection.pop(0)
            state,direction= gazeDetection(image)
            if state:
                print(direction)
                msg.append(direction)
                finalResult.append(image)
                continue
            print(direction)
            after_gaze_detection.append(image)
        if is_process_completed:
            break

def task5():
    global is_process_completed
    while True:
        if len(after_gaze_detection):
            image = after_gaze_detection[0]
            after_gaze_detection.pop(0)
            state,head_direction= headPosedetection(image)
            if state:
                print(head_direction)
                msg.append(head_direction)
                finalResult.append(image)
                continue
            print(head_direction)
            msg.append("SAFE")
            finalResult.append(image)
        if is_process_completed:
            break

def display():
    global is_process_completed
    while True:
        if len(finalResult):
            image = finalResult.pop(0)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                is_process_completed = True
                break
            image = cv2.resize(image, (800, 500))
            text = "SAFE"
            if msg:
                text=msg.pop(0)
            cv2.putText(image,text, (60, 60), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 0, 0), 2)
            cv2.imshow("face", image)
    is_process_completed = True

if __name__ == "__main__":
    t0 = threading.Thread(target=task0, name='capture')
    t0.start()
    t1 = threading.Thread(target=task1, name='mobile detection')
    t1.start()
    t2 = threading.Thread(target=task2, name='face detection')
    t2.start()
    t3 = threading.Thread(target=task3, name='spoof detection')
    t3.start()
    t4 = threading.Thread(target=task4, name='Gaze detection')
    t4.start()
    t5 = threading.Thread(target=task5, name='head pose  estimation')
    t5.start()
    display()

