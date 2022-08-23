import cv2

#cascade_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml")

#cascade_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")
#cascade_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
#cascade_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_lefteye_2splits.xml")
cascade_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_righteye_2splits.xml")
#cascade_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
#cascade_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
 
cap = cv2.VideoCapture(0)

while True:

    ret,frame = cap.read()

    gray = cv2.cvtColor(frame, 0)

    detections = cascade_classifier.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=5)

    if(len(detections) > 0):
        (x,y,w,h) = detections[0]
        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)                                                                                                                                                             


    
    cv2.imshow('detection',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):

        break

cap.release()
cv2.destroyAllWindows()