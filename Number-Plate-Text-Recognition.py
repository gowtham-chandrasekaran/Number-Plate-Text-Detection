#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  4 14:08:23 2021

@author: gowtham
"""
import cv2
import pytesseract #Need to install tesseract using - brew install tesseract

# Read the image file
image = cv2.imread('car1.jpeg')
cv2.imshow("Original",image)
cv2.waitKey(0)
# Convert to Grayscale Image
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#Canny Edge Detection
canny_edge = cv2.Canny(gray_image, 170, 200)

# Find contours based on Edges
contours, new  = cv2.findContours(canny_edge.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours=sorted(contours, key = cv2.contourArea, reverse = True)[:30] #Taking only top 30 contours of higher area

# Initialize license Plate contour and x,y coordinates
contour_with_license_plate = None
license_plate = None
x = None
y = None
w = None
h = None

#To Display all detected contours
image1 = image.copy()
cv2.drawContours(image1,contours,-1,(0,255,0),3)
cv2.imshow("All contours",image1)
cv2.waitKey(0)

# Find the contour with 4 potential corners and create ROI around it
for contour in contours:
        # Find Perimeter of contour and it should be a closed contour
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True) #0.03 is the accuracy percentage we are expecting
        if len(approx) == 4: #see whether it is a Rectangle
            contour_with_license_plate = approx
            x, y, w, h = cv2.boundingRect(contour)
            license_plate = gray_image[y:y + h, x:x + w]
            break

# Removing Noise from the detected image, before sending to Tesseract
license_plate = cv2.bilateralFilter(license_plate, 11, 17, 17)
(thresh, license_plate) = cv2.threshold(license_plate, 150, 180, cv2.THRESH_BINARY)

#Showing Number plate image
cv2.imshow("Cropped License Plate",license_plate)
cv2.waitKey(0)

#Text Recognition
# text = pytesseract.image_to_string(license_plate)
text = pytesseract.image_to_string(license_plate, config="--psm 7")

# Now we need to remove all special characters from the text to only have alphanumeric text
final_text = ""
for char in text:
    if(char.isalnum() == True):
        final_text+=char

#Draw License Plate and write the Text
image = cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 3)
image = cv2.putText(image, final_text, (x-100,y+150), cv2.FONT_HERSHEY_SIMPLEX, 3, (102,0,204), 6, cv2.LINE_AA)
print("License Plate :", final_text)

cv2.imshow("License Plate Detection",image)
cv2.waitKey(0)
