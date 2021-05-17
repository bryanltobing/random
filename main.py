import cv2 as cv
import caer

img = cv.imread('./dataset/testing_images/bryan-1.jpg')
resized = caer.resize(img, target_size=(400,400), preserve_aspect_ratio=True)
cv.imshow('Detected Faces', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray People', gray)

haar_cascade = cv.CascadeClassifier('./cascade/cascade.xml')

faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1)

print(f'Number of faces found = {len(faces_rect)}')

for (x,y,w,h) in faces_rect:
    cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)

cv.imshow('Detected Faces', img)



cv.waitKey(0)