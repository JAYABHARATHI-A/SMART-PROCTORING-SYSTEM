import cv2
import numpy as np
ls=["NO OF PEOPLE".ljust(14)+":","MOBILE FOUND".ljust(14)+":","LIVENESS".ljust(16)+":","LOOKING".ljust(16)+":"]
cap = cv2.VideoCapture(0)


#
# fourcc = cv2.VideoWriter_fourcc(*'MPEG')
# out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

def display(image,msg=0):
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return
    image = cv2.resize(image, (500, 300))
    image1 = cv2.resize(image, (500, 200))

    w_color = (134, 213, 250)
    if msg==0:
        msg=["one person","NO","HUMAN","STRAIGHT"]
        w_color = (255, 218, 148)
    cv2.rectangle(image1, (0, 0), (500,200), w_color, -1)
    x,y=10,30
    for i in range(len(msg)):
        cv2.putText(image1, ls[i]+msg[i], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        y+=50
    numpy_vertical = np.vstack((image1, image))
    numpy_vertical=cv2.resize(numpy_vertical, (400, 400))
    cv2.imshow("proctoring window", numpy_vertical)
    result.write(numpy_vertical)
#
result = cv2.VideoWriter('input.avi', cv2.VideoWriter_fourcc(*'MJPG'),25, (500,300))
flag=0
while True:
    ret, image = cap.read()
    if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
    image = cv2.resize(image, (500,300))
    # if flag:
    #     print("recording")
    result.write(image)
    cv2.imshow("proctoring window", image)

    # display(image)

cap.release()
result.release()
cv2.destroyAllWindows()

#
#
# import cv2
#
# cap = cv2.VideoCapture("output.avi")
# length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# fps = cap.get(cv2.CAP_PROP_FPS)
#
# print( length,fps )