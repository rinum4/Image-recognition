
"""
Распознавание ценников в1
"""

import math
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from pathlib import Path
import cv2 as cv
import pytesseract
from PIL import Image

class imgProc (BaseEstimator, ClassifierMixin):
    file = 'ikea'
    directory = str(Path()) + '/images/'
    # str(Path().absolute()) + '\\images\\'

    imgSave = []

    blue = (255,0,0)
    red = (0,0,255)
    green = (0,255,0)
    white = (255,255,255)

    lowerRed = (0,100,100)
    upperRed = (10,255,255)

    minArea = 20000

    minAreaTicket = 1000
    maxAreaTicket = 20000

    cannyFrom = 75
    cannyTo = 200

    dilate = 1

    arcLength = 0.05
    
    @staticmethod
    def pointOrder(point):
        ordered = [0,0,0,0]

        sum = [0,0,0,0]
        for x in range(len(point)):
            sum[x] = point[x,0,0]+point[x,0,1] # point[x].x+point[x].y
                    
        ordered[0] = point[sum.index(min(sum))]
        ordered[2] = point[sum.index(max(sum))]

        diff = [0,0,0,0]
        for x in range(len(point)):
            diff[x] = point[x,0,0]-point[x,0,1] # point[x].x-point[x].y
        
        ordered[1] = point[diff.index(max(diff))]
        ordered[3] = point[diff.index(min(diff))]

        return ordered
    
    @staticmethod
    def pointWidth(p1, p2):
        width = 0
        width = math.sqrt(math.pow((p2[0,0]-p1[0,0]), 2) + math.pow((p2[0,1]-p1[0,1]), 2))
        # Math.sqrt(Math.pow((p2.x-p1.x), 2) + Math.pow((p2.y-p1.y), 2))
        return width
        
    def saveImage(self,img):
        imgName = self.directory + self.file + str(len(self.imgSave)+1) + '.jpg'
        cv.imwrite(imgName, img)
        self.imgSave.append(imgName)
    
    def __init__(self):
        self.imgSave.append(self.directory+self.file+'.jpg')
    
imgProc =  imgProc()

img = cv.imread(imgProc.directory+imgProc.file+'.jpg')

# Convert BGR to HSV - перевод из BGR в HSV
process = cv.cvtColor(img, cv.COLOR_BGR2HSV)
             
# find only red - inRange - выделение красного цвета
process = cv.inRange(process, imgProc.lowerRed, imgProc.upperRed)
#imgProc.saveImage(process)

# Canny Edge detector - нахождение граней (вроде не особо нужно)
#process = cv.Canny(process, imgProc.cannyFrom, imgProc.cannyTo)
#imgProc.saveImage(process)

# dilate - расширение - ошибка !!! (вроде не особо нужно)
# process = cv.dilate(process, imgProc.dilate)
# imgProc.saveImage(process)

# find top contours - поиск контуров
contourImg = img

possibleContour = []  # массив для найденных контуров по нашим условиям
contours, hierarchy = cv.findContours(process, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

for i in range(len(contours)):
        # исключаем мелкие зоны imgProc.minArea
        if (cv.contourArea(contours[i]) < imgProc.minArea):
            continue
        
        #print(cv.contourArea(contours[i]))
        arcLength = cv.arcLength(contours[i], True)
        # «сглаживаем углы» найденных контуров, устраняя «шум
        contours[i] = cv.approxPolyDP(contours[i], imgProc.arcLength * arcLength, True)

        # cornerCount(i)
        if (len(contours[i])==4):
            # в случае нахождения контура из 4-ёх точек делаем им зелёный бордер
            cv.drawContours(contourImg, contours, i, imgProc.green, 3)
            # кладём контуры из 4-ёх точек в подготовленный массив
            possibleContour.append(imgProc.pointOrder(contours[i]))
            break # выходим т.к. нужна только 1 область
        else:
            # иначе красный бордер
            cv.drawContours(contourImg, contours, i, imgProc.red, 3)
                
#imgProc.saveImage(contourImg)

# выравниваем перспективу
warpImg = []
for x in range(len(possibleContour)):
    point = possibleContour[x]
    maxWidth = 0
    maxHeight = 0
    tmp = 0

    if (imgProc.pointWidth(point[0], point[1]) > imgProc.pointWidth(point[3], point[2])): 
        maxWidth = round(imgProc.pointWidth(point[0], point[1]))
    else:
        maxWidth = round(imgProc.pointWidth(point[3], point[2]))
    
    if (imgProc.pointWidth(point[0], point[3]) > imgProc.pointWidth(point[1], point[2])):
        maxHeight = round(imgProc.pointWidth(point[0], point[3]))
    else:
        maxHeight = round(imgProc.pointWidth(point[1], point[2]))
    
    tmpWarpImg = img
    # нужен UMat
    srcWarp = np.float32([[point[0][0,0], point[0][0,1]],
                          [point[1][0,0], point[1][0,1]],
                          [point[2][0,0], point[2][0,1]], 
                          [point[3][0,0], point[3][0,1]]])
    dstWarp = np.float32([[0, 0], [maxWidth, 0], [maxWidth, maxHeight], [0, maxHeight]])

    perspective = cv.getPerspectiveTransform(srcWarp, dstWarp)

    #cv.warpPerspective(tmpWarpImg, perspective, maxWidth, maxHeight, [255, 255, 255])
    tmpWarpImg = cv.warpPerspective(tmpWarpImg, perspective, (maxWidth, maxHeight))

    #повернем если необходимо 
    if (maxWidth < maxHeight):
        cv.rotate(tmpWarpImg,90)
    
    warpImg.append(tmpWarpImg)
    #imgProc.saveImage(tmpWarpImg)
          
# Ищем только то, что нужно (зону с 3мя четырехугольниками)
trueWarpImg = []
for x in range(len(warpImg)):
    warpedImg = warpImg[x]
    # Convert BGR to HSV - перевод из BGR в HSV
    warpedImg = cv.cvtColor(warpedImg, cv.COLOR_BGR2HSV)
    # выделение красного цвета
    warpedImg = cv.inRange(warpedImg, imgProc.lowerRed, imgProc.upperRed)
    #imgProc.saveImage(warpedImg)
    
    possibleContour = []
    contourImg = warpImg[x]
    #mode: RETR_EXTERNAL, RETR_LIST, RETR_CCOMP, RETR_TREE, RETR_FLOODFILL
    #ContourApproximationModes: 
    # CHAIN_APPROX_NONE 
    # CHAIN_APPROX_SIMPLE
    # CHAIN_APPROX_TC89_L1 
    # CHAIN_APPROX_TC89_KCOS
    contours, hierarchy = cv.findContours(warpedImg, cv.RETR_LIST, cv.CHAIN_APPROX_TC89_L1)
    for i in range(len(contours)):
        if (cv.contourArea(contours[i]) < 2000 or cv.contourArea(contours[i]) > 20000):
            continue # в этот раз ищем более мелкие области
    
        arcLength = cv.arcLength(contours[i], True)
        # «сглаживаем углы» найденных контуров, устраняя «шум
        contours[i] = cv.approxPolyDP(contours[i], imgProc.arcLength * arcLength, True)
    
        # cornerCount(i)
        if (len(contours[i])==4):
            # в случае нахождения контура из 4-ёх точек делаем им зелёный бордер
            cv.drawContours(contourImg, contours, i, imgProc.green, 3)
            # кладём контуры из 4-ёх точек в подготовленный массив
            possibleContour.append(imgProc.pointOrder(contours[i]))
            #break # не выходим т.к. нужны 3 области
        else:
            # иначе красный бордер
            cv.drawContours(contourImg, contours, i, imgProc.red, 3)
            
    #imgProc.saveImage(contourImg)
        
    # лишь в случае, если найдено 3 прямоугольника, мы продолжаем обработку
    if (len(possibleContour) == 3):
        trueContour = [-1,-1,-1]
        width = []
        tmpContour = possibleContour
    
        # сортируем найденные контуры. Сначала код товара, потом ряд и место. 
        for x2 in range(len(tmpContour)):
            width.append(tmpContour[x2][1][0,0] - tmpContour[x2][0][0,0])
            
        maxIndex = width.index(max(width))
        trueContour[0] = tmpContour[maxIndex]
    
        left = []
        for x2 in range(len(tmpContour)):
            if (x2 == maxIndex):
                continue
            left.append(tmpContour[x2][0][0,0])
            
        trueContour[1] = tmpContour[left.index(min(left))]
        trueContour[2] = tmpContour[left.index(max(left))]
    
        trueWarpImg.append({'img': warpImg[x], 'contour': trueContour})
        
# Для того, чтобы упростить распознавание, обрежем наши изображения, вычленим
# красный (дабы сделать изображение чб) и увеличим
labelImg = []
for x in range(len(trueWarpImg)):
    #labelImg[x] = []
    labelImg.append([])
    ticketImg = trueWarpImg[x]['img']
    ticketContour = trueWarpImg[x]['contour']

    for x2 in range(len(ticketContour)):
        point = ticketContour[x2]

        maxWidth = 0
        maxHeight = 0

        if (imgProc.pointWidth(point[0], point[1]) > imgProc.pointWidth(point[3], point[2])):
            maxWidth = round(imgProc.pointWidth(point[0], point[1]))
        else:
            maxWidth = round(imgProc.pointWidth(point[3], point[2]))
        
        if (imgProc.pointWidth(point[0], point[3]) > imgProc.pointWidth(point[1], point[2])):
            maxHeight = round(imgProc.pointWidth(point[0], point[3]))
        else:
            maxHeight = round(imgProc.pointWidth(point[1], point[2]))
        
        tmpWarpImg = ticketImg

        # нужен UMat
        srcWarp = np.float32([[point[0][0,0], point[0][0,1]],
                          [point[1][0,0], point[1][0,1]],
                          [point[2][0,0], point[2][0,1]], 
                          [point[3][0,0], point[3][0,1]]])
        dstWarp = np.float32([[0, 0], [maxWidth, 0], [maxWidth, maxHeight], [0, maxHeight]])

        perspective = cv.getPerspectiveTransform(srcWarp, dstWarp)

        #tmpWarpImg.warpPerspective(perspective, maxWidth, maxHeight, [255, 255, 255])
        tmpWarpImg = cv.warpPerspective(tmpWarpImg, perspective, (maxWidth, maxHeight))

        # crop
        # tmpWarpImg = tmpWarpImg.crop(2,2,tmpWarpImg.width()-4,tmpWarpImg.height()-4)
        height, width, channels = tmpWarpImg.shape
        tmpWarpImg = tmpWarpImg[2:height-4,2:width-4]

        labelImg[x].append(tmpWarpImg)
        #imgProc.saveImage(tmpWarpImg)
    
# распознаем
for x in range(len(labelImg)):
    label = labelImg[x]
    for x2 in range(len(label)):
        labelLine = label[x2]

        # Convert BGR to HSV - перевод из BGR в HSV
        labelLine = cv.cvtColor(labelLine, cv.COLOR_BGR2HSV)
        
        # выделение красного цвета
        labelLine = cv.inRange(labelLine, imgProc.lowerRed, imgProc.upperRed)

        # gaussianBlur([5,5]) # для большего эффекта можно провести Гаусс размытие
        #labelLine = cv.gaussianBlur(labelLine,[5,5])
        
        # увеличиваем изображение
        #height, width = labelLine.shape
        #labelLine = cv.resize(labelLine, (width*3,height*3))

        imgProc.saveImage(labelLine)
        
        #Tesseract.recognize(labelLine, {lang: 'eng',tessedit_char_whitelist: '0123456789.'})
        #res = Image.fromarray(labelLine)
        #res.save('out.jpg')
        print(pytesseract.image_to_string(Image.fromarray(labelLine),config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789.'))
        
    









             
             