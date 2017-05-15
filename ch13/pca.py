# -*- coding: utf-8 -*-
from numpy import *

def loadDataSet(fileName,delim='\t'):
    fr=open(fileName)
    #使用两个list来构建矩阵
    stringArr=[line.strip().split(delim) for line in fr.readlines()]
    datArr=[list(map(float,line)) for line in stringArr]
    return mat(datArr)

def pca(dataMat,topNfeat=9999999): #topNfeat为可选参数，记录特征值个数
    meanVals=mean(dataMat,axis=0) #求均值
    meanRemoved=dataMat-meanVals  #归一化数据
    covMat=cov(meanRemoved,rowvar=0)    #求协方差
    eigVals,eigVects=linalg.eig(mat(covMat)) #计算特征值和特征向量
    eigValInd=argsort(eigVals)               #对特征值进行排序，默认从小到大
    eigValInd=eigValInd[:-(topNfeat+1):-1]   #逆序取得特征值最大的元素
    redEigVects=eigVects[:,eigValInd]        #用特征向量构成矩阵
    lowDDataMat=meanRemoved*redEigVects      #用归一化后的各个数据与特征矩阵相乘，映射到新的空间
    reconMat=(lowDDataMat*redEigVects.T)+meanVals #还原原始数据
    return lowDDataMat,reconMat

#将NaN替换成平均值
def replaceNanWithMean():
    datMat=loadDataSet('secom.data',' ')
    numFeat=shape(datMat)[1]
    for i in range(numFeat):
        meanVal=mean(datMat[nonzero(~isnan(datMat[:,i].A))[0],i])
        datMat[nonzero(isnan(datMat[:,i].A))[0],i]=meanVal
    return datMat
















