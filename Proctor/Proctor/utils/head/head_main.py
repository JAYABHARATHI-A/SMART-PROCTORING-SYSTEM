#!/usr/bin/env python3
# python head_pose_from_webcam.py -f 1 -s 0
import cv2
import dlib
import numpy as np
import mediapipe as mp
from utils.head import reference_model as refer

# MediaPipe

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
PREDICTOR_PATH = "utils/model/shape_predictor_68_face_landmarks.dat"
# parser = argparse.ArgumentParser()
# parser.add_argument("-f", "--focal", type=float, help="Callibrated Focal Length of the camera")
# parser.add_argument("-s", "--camsource", type=int, default=0, help="Enter the camera source")
# args = vars(parser.parse_args())

face3Dmodel = refer.ref3DModel()


def main():
    predictor = dlib.shape_predictor(PREDICTOR_PATH)
    cap = cv2.VideoCapture(0)
    while True:
        GAZE = "Face Not Found"
        ret, img = cap.read()
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, c = image.shape
        image.flags.writeable = False
        results = face_detection.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if not ret:
            print(f'[ERROR - System]Cannot read from source:')
            break
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
            # cv2.rectangle(image, (absx, absy), (abswidth, absheight), (0, 255, 0), 2)
            shape = predictor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), newrect)
            refImgPts = refer.ref2dImagePoints(shape)
            height, width, channels = img.shape
            # focalLength = args["focal"] * width
            focalLength = 1 * width
            cameraMatrix = refer.cameraMatrix(focalLength, (height / 2, width / 2))
            mdists = np.zeros((4, 1), dtype=np.float64)

            # calculate rotation and translation vector using solvePnP
            success, rotationVector, translationVector = cv2.solvePnP(face3Dmodel, refImgPts, cameraMatrix, mdists)

            # noseEndPoints3D = np.array([[0, 0, 1000.0]], dtype=np.float64)
            # noseEndPoint2D, jacobian = cv2.projectPoints(noseEndPoints3D, rotationVector, translationVector,cameraMatrix, mdists)

            #  Nose line
            # p1 = (int(refImgPts[0, 0]), int(refImgPts[0, 1]))
            # p2 = (int(noseEndPoint2D[0, 0, 0]), int(noseEndPoint2D[0, 0, 1]))
            # cv2.line(image, p1, p2, (110, 220, 0), thickness=2, lineType=cv2.LINE_AA)

            # Euler angles
            rmat, jac = cv2.Rodrigues(rotationVector)
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
            # print('*' * 80)
            # print(f"Qx:{Qx}\tQy:{Qy}\tQz:{Qz}\t")
            # x = np.arctan2(Qx[2][1], Qx[2][2])
            # y = np.arctan2(-Qy[2][0], np.sqrt((Qy[2][1] * Qy[2][1]) + (Qy[2][2] * Qy[2][2])))
            # z = np.arctan2(Qz[0][0], Qz[1][0])
            # print("ThetaX: ", x)
            # print("ThetaY: ", y)
            # print("ThetaZ: ", z)
            # print('*' * 80)
            if angles[1] < -15:
                GAZE = "Looking: Left"+str(angles[1])
            elif angles[1] > 15:
                GAZE = "Looking: Right"+str(angles[1])

            else:
                GAZE = "Looking: Straight"+str(angles[1])

        cv2.putText(image, GAZE, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 80), 2)
        cv2.imshow("Head Pose", image)

        key = cv2.waitKey(10) & 0xFF
        if key == 27:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
