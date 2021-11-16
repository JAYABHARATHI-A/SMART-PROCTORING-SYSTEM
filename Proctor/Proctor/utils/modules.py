
import imutils as imutils
from utils.gaze.gaze_tracking import GazeTracking
import cv2
import dlib
import numpy as np
import mediapipe as mp
from utils.head import reference_model as refer
from tensorflow.keras.models import model_from_json


#face detection with dlib hog and svm
def face_detection(image):
    hog_face_detector = dlib.get_frontal_face_detector()
    image = imutils.resize(image, width=600)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces_hog = hog_face_detector(rgb, 0)
    return len(faces_hog),faces_hog

#mobile detection with YOLOv3 trained on coco dataset

def mobile_detection(img,net):

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    img = cv2.resize(img, None, fx=0.4, fy=0.4)
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)
    mobile_found = False

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            if class_id == 67:
                mobile_found = True
                return mobile_found
    return mobile_found





def gazeDetection(frame):
    gaze = GazeTracking()
    gaze.refresh(frame)
    # new_frame = gaze.annotated_frame()

    text = ""
    if gaze.is_right():
        text = "Looking right"
        res=False
    elif gaze.is_left():
        text = "Looking left"
        res=False
    elif gaze.is_center():
        text = "Looking center"
        res=True

    return res,text




def headPosedetection(image):

    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
    PREDICTOR_PATH = "utils/model/shape_predictor_68_face_landmarks.dat"
    face3Dmodel = refer.ref3DModel()
    predictor = dlib.shape_predictor(PREDICTOR_PATH)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, c = image.shape
    image.flags.writeable = False
    results = face_detection.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
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
        focalLength = 2 * width
        cameraMatrix = refer.cameraMatrix(focalLength, (height / 2, width / 2))
        mdists = np.zeros((4, 1), dtype=np.float64)

        # calculate rotation and translation vector using solvePnP
        success, rotationVector, translationVector = cv2.solvePnP(face3Dmodel, refImgPts, cameraMatrix, mdists)

        noseEndPoints3D = np.array([[0, 0, 1000.0]], dtype=np.float64)
        noseEndPoint2D, jacobian = cv2.projectPoints(noseEndPoints3D, rotationVector, translationVector,
                                                     cameraMatrix, mdists)

        # Euler angles
        rmat, jac = cv2.Rodrigues(rotationVector)
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

        if angles[1] < -15:
            GAZE = "Looking: Left"
            return False,GAZE
        elif angles[1] > 15:
            GAZE = "Looking: Right"
            return False,GAZE
        return True,"Looking : Straight"




def spoof_detection(face,frame):
    json_file = open('utils/model/project_antispoofing_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights('utils/model/project_antispoofing_model.h5')

    x = face.left()
    y = face.top()
    w = face.right() - x
    h = face.bottom() - y

    face = frame[y - 5:y + h + 5, x - 5:x + w + 5]
    resized_face = cv2.resize(face, (160, 160))
    resized_face = resized_face.astype("float") / 255.0
    resized_face = np.expand_dims(resized_face, axis=0)
    preds = model.predict(resized_face)[0]
    print(preds)
    if preds >= 0.02:
        label = 'spoof'
        return True
    else:
        label = 'real'
        return False





