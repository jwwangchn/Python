# -*- coding: UTF-8 -*-
import time

def BubbleSorting(_Array):
    "冒泡排序算法"
    swapped = True
    indexOfLastUnsortedElement = len(_Array)-1
    while swapped == True:
        swapped = False
        for i in range(0, indexOfLastUnsortedElement):
            if _Array[i] > _Array[i + 1]:
                _Array[i], _Array[i + 1] = _Array[i + 1], _Array[i]
                swapped = True
            #print _Array #打印排序过程
        indexOfLastUnsortedElement = indexOfLastUnsortedElement - 1
    return(_Array)


def SelectSorting(_Array):
    "选择排序"
    numOfElement = len(_Array)
    numOfUnsortedElement = numOfElement
    for i in range(0, numOfElement):
        currentMinOfElement = _Array[i]
        indexOfcurrentMin = i
        for j in range(numOfElement - numOfUnsortedElement, numOfElement):
            if currentMinOfElement > _Array[j]:
                currentMinOfElement = _Array[j]
                indexOfcurrentMin = j

        numOfUnsortedElement = numOfUnsortedElement-1
        _Array[i], _Array[indexOfcurrentMin] = _Array[indexOfcurrentMin], _Array[i]
    return(_Array)


def InsertSorting(_Array):
    "插入排序算法"
    numOfElement = len(_Array)
    lastSortedIndex = 0
    for i in range(1, numOfElement):
        insertFlag = False
        extractElement = _Array[lastSortedIndex + 1]
        for j in range(lastSortedIndex, -1, -1):
            currentSortedElement = _Array[j]
            if currentSortedElement > extractElement:
                _Array[j + 1] = _Array[j]
                indexOfInsert = j
                insertFlag = True
        if insertFlag:
            _Array[indexOfInsert] = extractElement
        lastSortedIndex = lastSortedIndex + 1
    return(_Array)

def PartitionOfQuickSorting(_Array, _left, _right, _pivotIndex):
    "快速排序分区算法"
    pivotValue = _Array[_pivotIndex]
    storeIndex = _left
    _Array[_pivotIndex], _Array[_right] = _Array[_right], _Array[_pivotIndex]
    for i in range(_left, _right):
        if _Array[i] < pivotValue:
            _Array[storeIndex], _Array[i] = _Array[i], _Array[storeIndex]
            storeIndex = storeIndex + 1
    _Array[_right], _Array[storeIndex] = _Array[storeIndex], _Array[_right]
    return(storeIndex)

def QuickSorting(_Array, _left, _right):
    "快速排序算法主程序"
    pivotNewIndex = _left
    if _right > _left:
        pivotNewIndex = PartitionOfQuickSorting(_Array, _left, _right, _left)
        QuickSorting(_Array, _left, pivotNewIndex - 1)
        QuickSorting(_Array, pivotNewIndex + 1, _right)

# 主函数, 分别计算每种算法的排序结果和运行时间

Array = [3, 44, 38, 5, 47, 15, 36, 26, 27, 2, 46, 4, 19, 50, 48]
start = time.clock()
BubbleSorting(Array)
BubbleSortingTime = (time.clock() - start)
print "BubbleSorting:"
print "Result:", Array
print "Time used:", BubbleSortingTime

Array = [3, 44, 38, 5, 47, 15, 36, 26, 27, 2, 46, 4, 19, 50, 48]
start = time.clock()
SelectSorting(Array)
SelectSortingTime = (time.clock() - start)
print "SelectSorting:"
print "Result:", Array
print "Time used:", SelectSortingTime

Array = [3, 44, 38, 5, 47, 15, 36, 26, 27, 2, 46, 4, 19, 50, 48]
start = time.clock()
InsertSorting(Array)
InsertSortingTime = (time.clock() - start)
print "InsertSorting:"
print "Result:", Array
print "Time used:", InsertSortingTime


Array = [3, 44, 38, 5, 47, 15, 36, 26, 27, 2, 46, 4, 19, 50, 48]
start = time.clock()
QuickSorting(Array, 0, len(Array)-1)
InsertSortingTime = (time.clock() - start)
print "QuickSorting:"
print "Result:", Array
print "Time used:", InsertSortingTime
