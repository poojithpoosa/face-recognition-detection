import pandas as pd
import cv2
import numpy as np
import time
f=[]
def add_face():
    na=input("enter name?\n")
    names=open("names.csv","a+")
    df=pd.read_csv("names.csv",index_col=None)
    n=df.shape[0]
    camera = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    i=0
    while i<5:
        time.sleep(2)
        return_value, image = camera.read()
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(image_gray)
        print(faces)
        for (x,y,w,h) in faces:
            cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),1)
        cv2.rectangle(image, ((0,image.shape[0] -25)),(270, image.shape[0]), (255,255,255), -1)
        cv2.imshow('Image with faces',image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        ans=input("Is your face detect?(y/n)\n")
        if (len(faces)==1 and (ans=='y' or ans=="Y")):
            print('pic number :',i+1)
            (x, y, w, h) = faces[0]
            imagess=image_gray[y:y+w, x:x+h]
            cv2.imwrite("images\\"+na+"."+str(i)+'.jpg', imagess)
            names.write(na+","+str(n/5)+"\n")
            i=i+1 
    names.close()

def face_detection():
    camera = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    while(True):
    # Capture frame-by-frame
        ret, frame = camera.read()
    # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        #print("Number of faces detected: " + str(faces.shape[0]))
        if len(faces)!=0:
            for (x,y,w,h) in faces:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)
                cv2.rectangle(frame,((0,frame.shape[0] -25)),(270, frame.shape[0]), (255,255,255), -1)
                cv2.putText(frame, "Number of faces detected: " + str(faces.shape[0]), (0,frame.shape[0] -10), cv2.FONT_HERSHEY_TRIPLEX, 0.5,  (0,0,0), 1)
        # Display the resulting frame   
        cv2.imshow('face detection',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    camera.release()
    cv2.destroyAllWindows()

def recognition():
    camera = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    df=pd.read_csv("names.csv")
    X=df.iloc[:,[0]].values
    Y=df.iloc[:,[1]].values
    n=df.shape[0]
    m=df.shape[1]
    f=[]
    j=0
    for i in range(n):
        if(i%5==0):
            j=0
        dirr="images//"+str(X[i,0])+"."+str(j)+".jpg"
        image=cv2.imread(dirr)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        f.append(image_gray)
        j=j+1
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(f,Y.astype(int))
    face_recognizer.save('trainer/trainer.yml')
    while(True):
        return_value, image = camera.read()
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(image_gray)
        for(x,y,w,h) in faces:
            cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)
            id, confidence = face_recognizer.predict(image_gray[y:y+h,x:x+w])
    
            # Check if confidence is less them 100 ==> "0" is perfect match 
            if (confidence < 100):
                id =X[id+4]
                confidence = "  {0}%".format(round(100 - confidence))
            else:
                id = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))
            
            cv2.putText(image, str(id), (x+5,y-5),cv2.FONT_HERSHEY_TRIPLEX, 1, (255,255,255), 2)
            cv2.putText(image, str(confidence), (x+5,y+h-5), cv2.FONT_HERSHEY_TRIPLEX, 1, (255,255,0), 1)  
        
        cv2.imshow('camera',image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    camera.release()
    cv2.destroyAllWindows()

while(True):
    print("select any option:")
    print("1.Add Face")
    print("2.Face Detection")
    print('3.Recognition')
    print('4.Exit\n')
    option=int(input())
    if(option==1):
        add_face()
    elif(option==2):
        face_detection()
    elif(option==3):
        recognition()
    else:
        break
    
    