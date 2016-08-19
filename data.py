#coding=utf-8
import os
import numpy
import random
from PIL import Image

CODE='0123456789abcdefghijklmnopqrstuvwxyz'

class DataSet():
    def __init__(self):
        self.fileRoot="/home/gpu/tensorflow/tensorflow/ai-cap/pro_cap/captcha2/"
        self.count=0
        self.pic=os.listdir(self.fileRoot)
        self.code=[code for code in CODE]
	#print(len(self.pic))

    def labelArrTrans(self,labelList):
        labelArrs=[[],[],[],[],[]]
        for i in range(len(labelList)):
            indexList=[]
            for j in range(len(labelArrs)):
                indexList.append(self.code.index(labelList[i][j]))
            for t in range(len(labelArrs)):
                labelArr=numpy.zeros(36)
                labelArr[indexList[t]]=1.0
                labelArrs[t].append(labelArr)
        labelArrs=numpy.array(labelArrs)
        return labelArrs



    def readLabels(self,batchID):
        #labelList=self.pic[self.count:self.count+batchSize]
	#print(labelList)
	labelList=[]
	for i in range(len(batchID)):
	    labelList.append(self.pic[batchID[i]])
        #print(labelList)
        labelArrs=self.labelArrTrans(labelList)
        labelArrs.astype(numpy.float32)
        return labelArrs

    def readImgs(self,batchID):
        imgList=[]
        for i in range(len(batchID)):
            with Image.open(self.fileRoot+self.pic[batchID[i]]).convert("L") as image:
                image=numpy.array(image)
		#print(image.shape)
                imgArr=image.reshape(1,10000)
                imgList.extend(imgArr)
                self.count=self.count+1
        imgArrs=numpy.array(imgList)
        imgArrs=imgArrs.astype(numpy.float32)
        imgArrs=numpy.multiply(imgArrs,1.0/255.0)
        return imgArrs

    def nextBatch(self,batchSize):
	batchID=[]
	for i in range(batchSize):
	    tempID=(int)(random.uniform(0,len(self.pic)))
            batchID.append(tempID)
        labelArrs=self.readLabels(batchID)
        imgArrs=self.readImgs(batchID)
        batch=[]
        batch.append(imgArrs)
        batch.append(labelArrs)
        return batch

    def nextRandBatch(self):
	labelList=os.listdir('/home/gpu/tensorflow/tensorflow/ai-cap/predict_list/')
	labelArrs=self.labelArrTrans(labelList)
	labelArrs.astype(numpy.float32)
	imgList=[]
	with Image.open('/home/gpu/tensorflow/tensorflow/ai-cap/predict_list/'+labelList[0]).convert('L') as image:
	    image=numpy.array(image)
	    imgArr=image.reshape(1,10000)
	    imgList.extend(imgArr)
	imgArrs=numpy.array(imgList)
	imgArrs=imgArrs.astype(numpy.float32)
	imgArrs=numpy.multiply(imgArrs,1.0/255.0)
	batch=[]
	batch.append(imgArrs)
	batch.append(labelArrs)
	return batch

if __name__=="__main__":
    data=DataSet()
    batch=data.nextBatch(6)
    print(batch[0].shape)
    print(batch[1].shape)
    print(batch[1])



