#coding:utf8
import matplotlib
import matplotlib.pyplot as plt
from numpy import *
import operator
#训练数据
def createDataSet():
    group=array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels=['A','A','B','B']
    return group,labels

#kNN算法
def classify0(inX,dataSet,labels,k):
    #inX为输入向量，dataSet为训练样本集，labels为训练样本标签
    dataSetSize=dataSet.shape[0]  #训练样本集数据量大小
    diffMat=tile(inX,(dataSetSize,1))-dataSet   #向量运算
    sqDiffMat=diffMat**2     #矩阵的每一个元素进行平方运算
    sqDistance=sqDiffMat.sum(axis=1)
    distances=sqDistance**0.5
    sortedDistIndicies=distances.argsort()  #返回排序后元素索引
    classCount={}
    for i in range(k):
        voteIlabel=labels[sortedDistIndicies[i]]    #返回距离最近的几个元素的标签
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1
    sortedClassCount=sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

#文件转换程序
def file2matrix(filename):
    fr=open(filename)
    arrayOLines=fr.readlines()
    numberOfLines=len(arrayOLines)
    returnMat=zeros((numberOfLines,3))
    classLabelVector=[]
    index=0
    for line in arrayOLines:
        line=line.strip()
        listFromLine=line.split('\t')
        returnMat[index,:]=listFromLine[0:3]
        if listFromLine[-1] == 'largeDoses':
            classLabelVector.append(1)            
        elif listFromLine[-1] == 'smallDoses':
            classLabelVector.append(2)
        elif listFromLine[-1] == 'didntLike':
            classLabelVector.append(3)
        index+=1
    return returnMat,classLabelVector

group,labels=createDataSet()
inXLabels=classify0([0,0],group,labels,3)
print '输入数据的类别为:',inXLabels

def autoNorm(dataSet):
    minVals=dataSet.min(0)
    maxVals=dataSet.max(0)
    ranges=maxVals-minVals
    normDataSet=zeros(shape(dataSet))
    m=dataSet.shape[0]
    normDataSet=dataSet-tile(minVals,(m,1))
    normDataSet=normDataSet/tile(ranges,(m,1))
    return normDataSet,ranges,minVals

def datingClassTest():
    hoRatio=0.1
    datingDataMat,datingLabels=file2matrix('datingTestSet.txt')
    normMat,ranges,minVals=autoNorm(datingDataMat)
    m=normMat.shape[0]
    numTestVecs=int(m*hoRatio)
    errorCount=0.0
    for i in range(numTestVecs):
        classifierResult=classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print 'the classifier came back with: %d,the real answer is: %d' % (classifierResult,datingLabels[i])
        if classifierResult != datingLabels[i]:
            errorCount+=1.0
    print 'the total arror rate is: %f' % (errorCount/float(numTestVecs))

def classifyPerson():
    resultList=['not at all','in small doses','in large doses']
    percentTats=float(raw_input('percentage of time spent playing video games?'))
    ffMiles=float(raw_input('frequent flier miles earned per years?'))
    iceCream=float(raw_input('liters of ice cream consumed per years?'))
    datingDataMat,datingLabels=file2matrix('datingTestSet.txt')
    normMat,ranges,minVals=autoNorm(datingDataMat)
    inArr=array([ffMiles,percentTats,iceCream])
    classifierResult=classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
    print 'you will probably like this person:',resultList[classifierResult-1]



# #绘制散点图
# fig1=plt.figure()
# ax1=fig1.add_subplot(111)
# ax1.scatter(datingDataMat[:,1],datingDataMat[:,2],15*array(datingLabels),15*array(datingLabels))
# fig2=plt.figure()
# ax2=fig2.add_subplot(111)
# ax2.scatter(datingDataMat[:,0],datingDataMat[:,1],15*array(datingLabels),15*array(datingLabels))
# plt.show()
#datingClassTest()
classifyPerson()