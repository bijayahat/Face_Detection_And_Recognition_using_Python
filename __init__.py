import cv2
import numpy as np
import  pickle
import sqlite3
import tkSimpleDialog
from tk import *
import os
from PIL import *


Name = "null"
Id = "null"
#-------------------------------------------
def takeInput():
    global Name
    global Id

    faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cam = cv2.VideoCapture(0)

    Name = tkSimpleDialog.askstring("Name","Enter Name: ")
    Id = tkSimpleDialog.askinteger("Age","Enter Age: ")
    print (Name +" "+str(Id))

    insertOrUpdate(Id,Name)

    sampleNum = 0
    while (True):
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceDetect.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            sampleNum = sampleNum + 1;
            cv2.imwrite("dataSet/User." + str(Id) + "." + str(sampleNum) + ".jpg", gray[y:y + h, x:x + h])
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.waitKey(100)
        cv2.imshow("Faces", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        elif sampleNum > 20:
            break

    cam.release()
    cv2.destroyAllWindows()
#-------------------------------------------
def insertOrUpdate(Id, Name):
    conn = sqlite3.connect("FaceBase.db")
    cmd = "SELECT * FROM People WHERE ID=" + str(Id)
    cursor = conn.execute(cmd)
    isRecordExist = 0
    for row in cursor:
        isRecordExist = 1
    if (isRecordExist == 1):
        cmd = "UPDATE People SET Name=" + str(Name) + "WHERE ID=" + str(Id)
    else:
        cmd = "INSERT INTO People(ID,Name) Values(" + str(Id) + "," + str(Name) + ")"
    conn.execute(cmd)
    conn.commit()
    conn.close()

#----------------------------------------------

def getImagewithId(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    print (imagePaths)
    faces = []
    IDs = []
    for imagePath in imagePaths:
        token = imagePath.split(".")
        if (token[-1] == 'jpg'):
            faceImg = Image.open(imagePath).convert('L')
            #print (imagePath)
            faceNp = np.array(faceImg,'float32')
            ID = int(os.path.split(imagePath)[-1].split(".")[1])
            faces.append(faceNp)
            IDs.append(ID)
            cv2.imshow("training",faceNp)
        #cv2.waitkey(10)
    return IDs,faces

#-----------------------------------------------
def train_image():
    recognizer = cv2.createLBPHFaceRecognizer()
    path = 'dataSet'
    IDs, faces = getImagewithId(path)
    recognizer.train(faces, np.array(IDs))
    recognizer.save('recognizer/trainningData.yml')
#-------------------------------------------------
def getProfile(id):
    conn = sqlite3.connect("FaceBase.db")
    cmd = "SELECT * FROM People WHERE ID=" + str(id)
    cursor = conn.execute(cmd)
    profile = None
    for row in cursor:
         profile = row
    conn.close()
    return profile

#-------------------------------------------------
def recognize():
    recognizer = cv2.createLBPHFaceRecognizer()
    recognizer.load('recognizer/trainningData.yml')
    cascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath);

    cam = cv2.VideoCapture(0)
    noo = input("enter user id")
    font = cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_SIMPLEX, 1, 1, 0, 1, 1)
    while True:
        ret, im = cam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x + w, y + h), (225, 0, 0), 2)
            Id, conf = recognizer.predict(gray[y:y + h, x:x + w])

            proflie = getProfile(Id)
            if conf < 60:
                if (proflie != None):
                    cv2.cv.PutText(cv2.cv.fromarray(im), str(proflie[0]), (x, y + h + 30), font, 255)
                    cv2.cv.PutText(cv2.cv.fromarray(im), str(proflie[2]), (x, y + h + 60), font, 255)
                    cv2.cv.PutText(cv2.cv.fromarray(im), str(proflie[3]), (x, y + h + 90), font, 255)
                    cv2.cv.PutText(cv2.cv.fromarray(im), str(proflie[4]), (x, y + h + 120), font, 255)
            else:
                cv2.cv.PutText(cv2.cv.fromarray(im), "Unknown", (x, y + h + 30), font, 255)
        cv2.imshow('im', im)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()


#--------------------------------------------------
root = Tk()
root.geometry("400x200")

button1 = Button(root,text="Take Image",command = takeInput)
button2 = Button(root,text="Train Image",command = train_image)
button3 = Button(root,text="Recognize ",command = recognize)

button1.pack(side='left')
button2.pack(side='left')
button3.pack(side='left')

root.mainloop()

