#!/bin/python3
import cv2

video_capture = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('detect.avi',fourcc, 20.0, (640,480))
facec= cv2.CascadeClassifier('cascades/haarcascade_profileface.xml')
face=cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
eye= cv2.CascadeClassifier('cascades/haarcascade_eye.xml')
smile=cv2.CascadeClassifier('cascades/haarcascade_smile.xml')

b=0
g=0
r=0

def detect(gray, frame):
    f=False
    faces= face.detectMultiScale(gray, 1.3, 3)
    facesr= facec.detectMultiScale(gray, 1.3, 3)
    facesl= facec.detectMultiScale(cv2.flip(gray,1), 1.3, 3)
    if len(faces) == 1:
        faced = faces
    elif len(facesr) == 1:
        faced=facesr
    elif len(facesl) == 1:
        faced=facesl
        f=True
    else:
        faced=()
    for (x, y, w, h) in faced:
        if f:
            frame=cv2.flip(frame,1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (b, g, r), 1)
        # cv2.circle(frame,(int(x+0.5*w),int(y+0.5*h)),max(w,h), (b,g,r),3)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye.detectMultiScale(roi_gray, 1.1, 18)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 1)
        smiles = smile.detectMultiScale(roi_gray, 1.7, 18)
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (150, 200, 121), 1)
        if f:
            frame=cv2.flip(frame,1)

        cv2.putText(frame,"FACE DETECTED!!!", (x+w,y+h), cv2.FONT_ITALIC,1, (2,200,25))
        out.write(frame)
    cv2.putText(frame,"Press Esc To Exit",(250,25),cv2.FONT_ITALIC,0.5,(20,20,20))
    return frame

def getRed(x):
    global r
    r= x
def getBlue(x):
    global b
    b= x
def getGreen(x):
    global g
    g= x
    
cv2.namedWindow("DETECT_FACE")
cv2.createTrackbar('R',"DETECT_FACE",0,255,getRed)
cv2.createTrackbar('B',"DETECT_FACE",150,255,getBlue)
cv2.createTrackbar('G',"DETECT_FACE",200,255,getGreen)





if not (face.empty() or eye.empty() or smile.empty() or facec.empty()):
   while True:

       _, frame = video_capture.read()
       frame=cv2.flip(frame,1)
       gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
       canvas = detect(gray, frame)
       cv2.imshow("DETECT_FACE", canvas)
       if cv2.waitKey(1) & 0xFF ==27:
           break


else:
    print("CHECK CASCADES NOT LOADED")
video_capture.release()
cv2.destroyAllWindows()

