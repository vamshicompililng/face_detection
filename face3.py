import cv2
# casade_classifier=cv2.CascadeClassifier('/Users/vamshiartham/Documents/face-dectation/haarcascade_frontalface_default.xml')
# casade_classifier=cv2.CascadeClassifier('/Users/vamshiartham/Documents/face-dectation/haarcascade_eye_tree_eyeglasses.xml')
# casade_classifier=cv2.CascadeClassifier('/Users/vamshiartham/Documents/face-dectation/haarcascade_eye.xml')
# casade_classifier=cv2.CascadeClassifier('/Users/vamshiartham/Documents/face-dectation/haarcascade_fullbody.xml')
# casade_classifier=cv2.CascadeClassifier('/Users/vamshiartham/Documents/face-dectation/haarcascade_lefteye_2splits.xml')
casade_classifier=cv2.CascadeClassifier('/Users/vamshiartham/Documents/face-dectation/haarcascade_smile.xml')
# casade_classifier=cv2.CascadeClassifier('/Users/vamshiartham/Documents/face-dectation/haarcascade_righteye_2splits.xml')
cap=cv2.VideoCapture(0)  #web cam capture


while True:
    # capture frame by frame
    ret,frame=cap.read()

    gray=cv2.cvtColor(frame,0)
    detections=casade_classifier.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5)
    if(len(detections)>0):
        (x,y,w,h)=detections[0]
        frame=cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)


        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break


cap.release()
cv2.destroyAllWindows()