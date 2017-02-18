#coding=utf-8
import operator
import matplotlib.pyplot as plt
#绘制属性图
decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")

#构造注解树 在python字典形式中如何存储树
def getNumLeafs(myTree):
    numLeafs=0 #初始化结点数
    # 下面三行为代码 python3 替换注释的两行代码
    firstSides = list(myTree.keys())
    firstStr = firstSides[0]  # 找到输入的第一个元素,第一个关键词为划分数据集类别的标签
    secondDict = myTree[firstStr]
    #firstStr = list(myTree)
    #secondDict=myTree[firstStr]
    for key in secondDict.keys(): #测试数据是否为字典形式
        if type(secondDict[key]).__name__=='dict': #type判断子结点是否为字典类型
            numLeafs+=getNumLeafs(secondDict[key])
            #若子节点也为字典，则也是判断结点，需要递归获取num
        else:  numLeafs+=1
    return numLeafs #返回整棵树的结点数
def getTreeDepth(myTree):
    maxDepth=0
    # 下面三行为代码 python3 替换注释的两行代码
    firstSides = list(myTree.keys())
    firstStr = firstSides[0]
    secondDict = myTree[firstStr]
    #firstStr=myTree.keys()[0]
    #secondDict=myTree[firstStr]#获取划分类别的标签
    for key in secondDict.keys():
        if type(secondDict[key]) == dict:
           thisDepth = 1 + getTreeDepth(secondDict[key])
    else:
        thisDepth = 1
    if thisDepth > maxDepth: maxDepth = thisDepth
    return maxDepth

def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction',
             xytext=centerPt, textcoords='axes fraction',
             va="center", ha="center", bbox=nodeType, arrowprops=arrow_args )

def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)

def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = getNumLeafs(myTree)  #计算树的宽度  totalW
    depth = getTreeDepth(myTree) #计算树的高度 存储在totalD
    #python3.x修改
    firstSides = list(myTree.keys())#firstStr = myTree.keys()[0]     #the text label for this node should be this
    firstStr = firstSides[0]  # 找到输入的第一个元素
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)#按照叶子结点个数划分x轴
    plotMidText(cntrPt, parentPt, nodeTxt) #标注结点属性
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD #y方向上的摆放位置 自上而下绘制，因此递减y值
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':#判断是否为字典 不是则为叶子结点
            plotTree(secondDict[key],cntrPt,str(key))        #递归继续向下找
        else:   #为叶子结点
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW #x方向计算结点坐标
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)#绘制
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))#添加文本信息
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD #下次重新调用时恢复y

def createPlot(inTree): #主函数
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)  # no ticks
    # createPlot.ax1 = plt.subplot(111, frameon=False) #ticks for demo puropses
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5 / plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()

#输出预先存储的树信息，避免每次测试都需要重新创建树
def retrieveTree(i):
    listOfTrees =[{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                  {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                  ]
    return listOfTrees[i]















