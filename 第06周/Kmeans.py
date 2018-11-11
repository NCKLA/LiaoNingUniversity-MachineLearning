from pandas import DataFrame,Series # 导入所需要的两个函数
import matplotlib.pyplot as plt
n = 10    # 待聚类样本个数
x = Series(range(1,n+1))      # 样本的x坐标值
y = Series(range(n,n+n))      # 样本的y坐标值
center0 = Series([x[0],y[0]]) # 初始第一个类中心
center1 = Series([x[1],y[1]]) # 初始第二个类中心
dis = DataFrame(index=range(0,n),columns=['dis_cen0','dis_cen1','class']) # 初始数据框
def which_min(x):
    if x[0] <= x[1]:
        z = 0
    else:
        z = 1
    return z
while True:
    for i in range(0, n):       # 求各样本至各类中心的距离
        dis.ix[i, 0] = ((x[i]-center0[0])**2+(y[i]-center0[1])**2)**0.5
        dis.ix[i, 1] = ((x[i]-center1[0])**2+(y[i]-center1[1])**2)**0.5
        dis.ix[i, 2] = which_min(dis.ix[i,0:2]) # 样本归类
    index0 = dis.ix[:, 2] == 0  # 第一类样本序号
    index1 = dis.ix[:, 2] == 1  # 第二类样本序号
    # ====求新类中心===
    center0_new = Series([x[index0].mean(),y[index0].mean()])
    center1_new = Series([x[index1].mean(),y[index1].mean()])
    # ====判定类中心是否发生变化,否则中止while循环====
    if sum(center0==center0_new)+sum(center1==center1_new)==4:
        break
    center0 = center0_new
    center1 = center1_new
print(dis) # 输出结果
