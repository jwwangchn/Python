#coding:utf8
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
    print '训练样本集大小:',dataSetSize
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
    returnMat=zeros((numberOfLines,3)
    classLabelVector=[]
    index=0
    for line in arrayOLines:
        line=line.strip()
        listFromLine=line.split('\t')
        returnMat[index,:]=listFromLine[0:3]
        classLabelVecor.append(int(listFromLine))

group,labels=createDataSet()
inXLabels=classify0([0,0],group,labels,3)
print '输入数据的类别为:',inXLabels
