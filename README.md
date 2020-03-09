The python code Contains three funtions for Adding new face,face detection, face recognition.
the program asks for 4 options:
1)add face
2)face detection
3)face recognition
4)exit
Add_face fucntion:
5 photos of the users is taken and saved in images folder(create it if not present).
while adding faces it asks users to say yes or no if his face was detected or not. the user name and label is saved in names.csv file.

face detection function:
this function detects the face and shows the green box around the face.

face recogntion function:
this function uses names.csv file and images in image folder to train the model and predict the user and display the confidence.

opencv, pandas, numpy libraries needed to run this code

