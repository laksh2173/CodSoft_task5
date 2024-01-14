import pathlib
import cv2

cascade_path = pathlib.Path(cv2.__file__).parent.absolute()/"data/haarcascade_frontalface_default.xml"

face_cascade = cv2.CascadeClassifier(str(cascade_path))
video_capture = cv2.VideoCapture(0)

while True:
    ret , video =video_capture.read()
    gray = cv2.cvtColor(video, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,minNeighbors=5,
        minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE
    )

    for(x,y,w,h) in faces:
        cv2.rectangle(video, (x,y), (x+w, y+h), (255,255,0) ,2)

    cv2.imshow("Video", video)
    if cv2.waitKey(1) == ord("l"):
        break

video_capture.release()
cv2.destroyAllWindows()