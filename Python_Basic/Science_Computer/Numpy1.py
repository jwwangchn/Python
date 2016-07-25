#coding:utf8
from numpy import *
v=array([1,2,3,4])
print v
M=array([[1,2],[3,4]])
print M
print type(v),type(M)
print v.shape,M.shape   #获取维度数据
print M.size            #获取元素个数
print M.dtype
M=array([[1,2],[3,4]],dtype=complex)    #直接定义数据类型
print M