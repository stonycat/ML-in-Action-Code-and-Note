#coding=utf-8
from numpy import *

def loadDataSet(fileName):
    numFeat=len(open(fileName).readline().split('\t'))-1
    dataMat=[]; labelMat=[]
    fr=open(fileName)
    for line in fr.readlines():
        lineArr=[]
        curLine=line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

def standRegres(xArr,yArr):
    xMat=mat(xArr); yMat=mat(yArr).T
    xTx=xMat.T*xMat
    if linalg.det(xTx)==0.0:  #矩阵行列式|A|=0,则矩阵不可逆
        print("This matrix is singular, cannot do inverse")
        return
    ws=xTx.I*(xMat.T*yMat)
    return ws  #回归系数w
#局部加权线性回归 LWLR
def lwlr(testPoint,xArr,yArr,k=1.0):
    xMat=mat(xArr); yMat=mat(yArr).T
    m=shape(xMat)[0]
    weights=mat(eye((m)))
    for j in range(m):
        diffMat = testPoint - xMat[j,:]   #计算样本点与预测值的距离
        weights[j,j] = exp(diffMat*diffMat.T/(-2.0*k**2)) #计算高斯核函数W
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0: #判断是否可逆
        print ("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws
#测试
def lwlrTest(testArr,xArr,yArr,k=1.0):  #loops over all the data points and applies lwlr to each one
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i],xArr,yArr,k)
    return yHat

def rssError(yArr,yHatArr):
    return((yArr-yHatArr)**2).sum()

#缩减系数之“岭”回归
def ridgeRegres(xMat,yMat,lam=0.2):
    xTx=xMat.T*xMat
    denom=xTx+eye(shape(xMat)[1])*lam
    if linalg.det(denom)==0.0:
        print("This matrix is singular,cannot do inverse")
        return
    ws=denom.I*(xMat.T*yMat)
    return ws
#测试
def ridgeTest(xArr,yArr):
    xMat = mat(xArr); yMat = mat(yArr).T#数据标准化（特征标准化处理），减去均值，除以方差
    yMean=mean(yMat,0)
    yMat=yMat-yMean
    xMeans=mean(xMat,0)
    xVar=var(xMat,0)
    xMat=(xMat-xMeans)/xVar
    numTestPts=30
    wMat=zeros((numTestPts,shape(xMat)[1]))
    for i in range(numTestPts):
        ws=ridgeRegres(xMat,yMat,exp(i-10))
        wMat[i,:]=ws.T
    return wMat

def regularize(xMat):#regularize by columns
    inMat = xMat.copy()
    inMeans = mean(inMat,0)   #calc mean then subtract it off
    inVar = var(inMat,0)      #calc variance of Xi then divide by it
    inMat = (inMat - inMeans)/inVar
    return inMat

#前向逐步线性回归
def stageWise(xArr,yArr,eps=0.01,numIt=100):
    xMat=mat(xArr); yMat=mat(yArr).T
    yMean=mean(yMat,0)
    yMat=yMat-yMean
    xMat=regularize(xMat)
    m,n=shape(xMat)
    returnMat=zeros((numIt,n))
    ws=zeros((n,1)); wsTest=ws.copy(); wsMax=ws.copy()
    for i in range(numIt):
        print (ws.T)
        lowestError=inf;
        for j in range(n):
            for sign in [-1,1]: #两次循环，计算增加或者减少该特征对误差的影响
                wsTest=ws.copy()
                wsTest[j]+=eps*sign
                yTest=xMat*wsTest
                rssE=rssError(yMat.A,yTest.A) #平方误差
                if rssE<lowestError:
                    lowestError=rssE
                    wsMax=wsTest
        ws=wsMax.copy()
        returnMat[i,:]=ws.T
    return returnMat


#使用google获取购物信息
def setDataCollect(retX, retY):
    scrapePage("setHtml/lego8288.html", "data/lego8288.txt", 2006, 800, 49.99)
    scrapePage("setHtml/lego10030.html", "data/lego10030.txt", 2002, 3096, 269.99)
    scrapePage("setHtml/lego10179.html", "data/lego10179.txt", 2007, 5195, 499.99)
    scrapePage("setHtml/lego10181.html", "data/lego10181.txt", 2007, 3428, 199.99)
    scrapePage("setHtml/lego10189.html", "data/lego10189.txt", 2008, 5922, 299.99)
    scrapePage("setHtml/lego10196.html", "data/lego10196.txt", 2009, 3263, 249.99)



