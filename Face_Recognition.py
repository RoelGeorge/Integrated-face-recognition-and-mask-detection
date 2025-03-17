import cv2
from face_recognition.api import face_locations
import numpy as np
import face_recognition
import os
from datetime import datetime


path="DataFolder"
images=[]
classNames=[]

mylist=os.listdir(path) 
#print(mylist)

for cl in mylist:
    curImg=cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findEncodings(images):
    encodeList=[]
    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList=f.readlines()
        nameList=[]
        for line in myDataList:
            entry=line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now=datetime.now()
            dtString=now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')
    


encodeListKnown=findEncodings(images)

print("Encoding Complete")

print("Initiating Video Capture...")
cap=cv2.VideoCapture(0)

while True:
    success,img = cap.read()
    imgS=cv2.resize(img,(0,0),None,0.25,0.25)
    imgS=cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)

    facesCurFrame=face_recognition.face_locations(imgS)
    encodesCurFrame=face_recognition.face_encodings(imgS,facesCurFrame)

    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
        matches=face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis=face_recognition.face_distance(encodeListKnown,encodeFace)
        print(faceDis)
        matchIndex=np.argmin(faceDis)

        if matches[matchIndex]:
            name=classNames[matchIndex].upper()
            print(name)
            y1,x2,y2,x1=faceLoc
            cv2.rectangle(img,(x1*4,y1*4),(x2*4,y2*4),(0,255,2),2)
            #cv2.rectangle(img,(x1*4,y1*4-35),(x2*4,y2*4),(0,255,2),cv2.FILLED)
            cv2.putText(img,name,(x1*4+6,y2*4-6),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),2)
            markAttendance(name)
    key = cv2.waitKey(1) & 0xFF 
    if key == ord("q"):
        break
    if key ==ord("m"):
        cap.release()
        cv2.destroyAllWindows()
        exec(open('detect_mask_video.py').read())
        break


    cv2.imshow("Webcam",img)
    cv2.waitKey(1)




