#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 17:41:15 2018

@author: yusufbasol
"""
import cv2

yuzCas= cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
gozCas= cv2.CascadeClassifier("haarcascade_eye.xml")


kamera= cv2.VideoCapture(0)
kamera.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
kamera.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

while True:
    _,goruntu=kamera.read()
   
    
    griton=cv2.cvtColor(goruntu,cv2.COLOR_BGR2GRAY)
    yuzler=yuzCas.detectMultiScale(griton,1.3,5)
    
    width = kamera.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = kamera.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    for(x,y,w,h) in yuzler:
        cv2.rectangle(goruntu,(x,y),(x+w,y+h), (255,0,0),2) # goruntu, sol üst kose, sağ alt kose, color, kalınlık 
        roiGri=griton[y:y+h,x:x+w]
        roiRenkli=goruntu[y:y+h,x:x+w]
        gozler=gozCas.detectMultiScale(roiGri)
        for(ex,ey,ew,eh) in gozler:
             cv2.rectangle(roiRenkli,(ex,ey),(ex+ew,ey+eh),(0,158,0),2)
    cv2.imshow("goruntu",goruntu)
    if cv2.waitKey(13) & 0xFF=="q":
        break
kamera.release()
cv2.destroyAllWindows()
