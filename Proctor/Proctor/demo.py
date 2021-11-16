import dlib
import cv2
import imutils as imutils
import mediapipe as mp
import numpy as np

from tensorflow.keras.models import model_from_json
from utils.gaze.gaze_tracking import GazeTracking
from utils.head import reference_model as refer

# def display(image,msg=0):
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         return False
#     image = cv2.resize(image, (800, 500))
#     text = "SAFE"
#     if msg:
#         text=msg
#     cv2.putText(image,text, (60, 60), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 0, 0), 2)
#     cv2.imshow("face", image)
#     return True

result = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'MJPG'),25, (500,500))


ls=["MOBILE FOUND".ljust(13)+": ","NO OF PEOPLE".ljust(13)+": ","LIVENESS".ljust(15)+": ","LOOKING".ljust(15)+": "]

def display(image,msg=[]):
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return False
    image = cv2.resize(image, (500, 300))
    image1 = cv2.resize(image, (500, 200))

    w_color = (134, 213, 250)
    t_color=(0,0,0)
    if len(msg)==0:
        msg=["NO","ONE PERSON","HUMAN","STRAIGHT"]
        w_color = (255, 218, 148)
        # t_color = (0,0,0)
    cv2.rectangle(image1, (0, 0), (500,200), w_color, -1)
    x,y=10,30
    for i in range(len(msg)):
        cv2.putText(image1, (ls[i]+msg[i]).upper(), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, t_color, 2)
        # print(ls[i],msg[i])
        y+=50
    both = np.vstack((image1, image))
    cv2.imshow("proctoring window", both)
    # both = cv2.vconcat([image, image1])
    image = cv2.resize(both, (500, 500))
    result.write(image)
    return  True







#loading the model for mobile detection
configPath = 'utils/model/yolov3.cfg'
weightsPath = 'utils/model/yolov3.weights'
net = cv2.dnn.readNet(weightsPath, configPath)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

#loading the model for spoof detection
json_file = open('utils/model/project_antispoofing_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights('utils/model/project_antispoofing_model.h5')

#loading the model for facial landmark detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
PREDICTOR_PATH = "utils/model/shape_predictor_68_face_landmarks.dat"
face3Dmodel = refer.ref3DModel()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

vid = cv2.VideoCapture("input/input.mp4")
frame_c=5
# vid.set(cv2.CAP_PROP_POS_MSEC,20*1000)
msg=[]
while True:
    #------------------------ capturing image from source------------------------------------------
    ret, image = vid.read()
    if not ret:
        is_process_completed = True
        break
    frame_c += 1
    if frame_c<5:
        print(msg)
        display(image, msg)
        continue
    frame_c=0
    msg=[]
    #------------------------ mobile detection------------------------------------------
    img = cv2.resize(image, None, fx=0.4, fy=0.4)
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)
    mobile_found = False
    flag=0
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            if class_id == 67:
                msg.append("mobile found")
                flag=1
                break
        if flag:
            break
    if flag:
        if display(image, msg):
            continue
        break
    #------------------------ face detection------------------------------------------
    msg.append("NO")
    hog_face_detector = dlib.get_frontal_face_detector()
    image = imutils.resize(image, width=600)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces_hog = hog_face_detector(rgb, 0)
    if len(faces_hog)==0:
        msg.append("no face")
        if display(image,msg):
            continue
        break

    elif len(faces_hog)>1:
        msg.append("Multiple faces found")
        if display(image,msg):
            continue
        break


    #------------------------ face spoofing detction-----------------------------------------
    msg.append("ONE PERSON")
    face=faces_hog[0]
    x = face.left()
    y = face.top()
    w = face.right() - x
    h = face.bottom() - y

    face = image[y - 5:y + h + 5, x - 5:x + w + 5]
    resized_face = cv2.resize(face, (160, 160))
    resized_face = resized_face.astype("float") / 255.0
    resized_face = np.expand_dims(resized_face, axis=0)
    preds = model.predict(resized_face)[0]
    if preds >= 0.2:
        label = 'spoof'
        msg.append("Spoof image")
        if display(image, msg):
            continue
        break


    #------------------------ Gaze detction------------------------------------------
    msg.append("HUMAN")
    gaze = GazeTracking()
    gaze.refresh(image)
    # new_frame = gaze.annotated_frame()
    if gaze.is_right():
        text = "right (GAZE)"
        msg.append(text)
        if display(image,msg):
            continue
        break
    elif gaze.is_left():
        text = "left (GAZE)"
        msg.append(text)
        if display(image,msg):
            continue
        break


    #------------------------ headpose estimation------------------------------------------

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, c = image.shape
    image.flags.writeable = False
    results = face_detection.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    try:
        if results.detections:
            for detection in results.detections:
                location = detection.location_data
                relative_bounding_box = location.relative_bounding_box
                x_min = relative_bounding_box.xmin
                y_min = relative_bounding_box.ymin
                width1 = relative_bounding_box.width
                height1 = relative_bounding_box.height
                absx, absy = mp_drawing._normalized_to_pixel_coordinates(x_min, y_min, w, h)
                abswidth, absheight = mp_drawing._normalized_to_pixel_coordinates(x_min + width1, y_min + height1, w, h)

            newrect = dlib.rectangle(absx, absy, abswidth, absheight)
            shape = predictor(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), newrect)
            refImgPts = refer.ref2dImagePoints(shape)
            height, width, channels = image.shape
            focalLength = 1 * width
            cameraMatrix = refer.cameraMatrix(focalLength, (height / 2, width / 2))
            mdists = np.zeros((4, 1), dtype=np.float64)

            success, rotationVector, translationVector = cv2.solvePnP(face3Dmodel, refImgPts, cameraMatrix, mdists)

            noseEndPoints3D = np.array([[0, 0, 1000.0]], dtype=np.float64)
            noseEndPoint2D, jacobian = cv2.projectPoints(noseEndPoints3D, rotationVector, translationVector,cameraMatrix, mdists)

            rmat, jac = cv2.Rodrigues(rotationVector)
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            if angles[1] < -15:
                # print(angles)
                GAZE = "Left (Head pose)"
                        # +str(angles[1])
                msg.append(GAZE)
                if display(image,msg):
                    continue
            elif angles[1] > 15:
                # print(angles)
                GAZE = "Right (Head pose)"
                    # +str(angles[1])
                msg.append(GAZE)
                if display(image, msg):
                    continue
    except:
        frame_c-=1
        continue
    if display(image):
        msg=[]
        continue
    break

vid.release()