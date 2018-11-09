import regression
from numpy import *
xArr,yArr=regression.loadDataSet('fileName.txt')
print(xArr[0:10])
ws=regression.standRegres(xArr,yArr)
print(ws)
xMat=mat(xArr)
yMat=mat(yArr)
yHat=xMat*ws
#绘制散点图和最佳拟合直线图
import matplotlib.pyplot as plt
fig=plt.figure()
ax=fig.add_subplot(111)
ax.scatter(xMat[:,1].flatten().A[0],yMat.T[:,0].flatten().A[0])
xCopy=xMat.copy()
xCopy.sort(0)
yHat=xCopy*ws
ax.plot(xCopy[:,1],yHat)
plt.show()
