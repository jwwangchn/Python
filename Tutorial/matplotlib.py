import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# 1. first demo
# x = np.linspace(-1, 1, 50)
# y = x**2 + 1
# plt.plot(x, y)
# plt.show()

# 2. second demo
# x = np.linspace(-3, 3, 50)
#
# y1 = x**2 + 1
# y2 = 2*x + 1
#
# plt.figure(num = 1, figsize=(8,5))
# plt.plot(x, y1)
# plt.plot(x, y2, color='red', linewidth=2.0, linestyle = '--')
#
#
# plt.figure(2)
# plt.plot(x, y2)
# plt.show()

# 3. third demo
# x = np.linspace(-3, 3, 50)
#
# y1 = x**2 + 1
# y2 = 2*x + 1
#
# plt.figure(1)
# plt.plot(x, y1)
# plt.plot(x, y2, color='red', linewidth=2.0, linestyle = '--')
# plt.xlim((-1, 2))
# plt.ylim((-2, 3))
# plt.xlabel('I am x')
# plt.ylabel('I am y')
# new_ticks = np.linspace(-1,2,5)
# print(new_ticks)
# plt.xticks(new_ticks)
# plt.yticks([0,-1.8,-2,1.22,3], [r'$\alpha$', '$b$', '$c$', '$d$', '$e$'])
#
#
# plt.show()

# 4. Modify the position of axis
# x = np.linspace(-3, 3, 50)
#
# y1 = x**2 + 1
# y2 = 2*x + 1
#
# plt.figure(1)
# plt.plot(x, y1)
# plt.plot(x, y2, color='red', linewidth=2.0, linestyle = '--')
# plt.xlim((-1, 2))
# plt.ylim((-2, 3))
# plt.xlabel('I am x')
# plt.ylabel('I am y')
# new_ticks = np.linspace(-1,2,5)
# print(new_ticks)
# plt.xticks(new_ticks)
# plt.yticks([0,-1.8,-3,1.22,3], [r'$\alpha$', r'$b$', r'$c$', r'$d$', r'$e$'])
#
# # gca = 'get current axis'
# ax = plt.gca()
# ax.spines['right'].set_color('none')
# ax.spines['top'].set_color('none')
# ax.xaxis.set_ticks_position('bottom')
# ax.yaxis.set_ticks_position('left')
# ax.spines['bottom'].set_position(('data', -1))
# ax.spines['left'].set_position(('data', 0))
#
# plt.show()

# 5. add legend
# x = np.linspace(-3, 3, 50)
#
# y1 = x**2 + 1
# y2 = 2*x + 1
#
# plt.figure(1)
# plt.plot(x, y1)
# plt.plot(x, y2, color='red', linewidth=2.0, linestyle = '--')
# plt.xlim((-1, 2))
# plt.ylim((-2, 3))
# plt.xlabel('I am x')
# plt.ylabel('I am y')
# new_ticks = np.linspace(-1,2,5)
# print(new_ticks)
# plt.xticks(new_ticks)
# plt.yticks([0,-1.8,-3,1.22,3], [r'$\alpha$', r'$b$', r'$c$', r'$d$', r'$e$'])
#
# # gca = 'get current axis'
# plt.figure(2)
# l1, = plt.plot(x, y2, label='up')
# l2, = plt.plot(x, y1, color='red', linewidth=2.0, linestyle = '--', label='down')
# plt.legend(handles = [l1, l2], labels = ['aaa', 'bbb'], loc = 'best')
# plt.show()

# 6. add commitment
# x = np.linspace(-3, 3, 50)
#
# y1 = x**2 + 1
# y2 = 2*x + 1
#
# plt.figure(1)
# plt.plot(x, y1)
# plt.plot(x, y2, color='red', linewidth=2.0, linestyle = '--')
# plt.xlim((-1, 2))
# plt.ylim((-2, 3))
# plt.xlabel('I am x')
# plt.ylabel('I am y')
# new_ticks = np.linspace(-1,2,5)
# print(new_ticks)
# # plt.xticks(new_ticks)
# # plt.yticks([0,-1.8,-3,1.22,3], [r'$\alpha$', r'$b$', r'$c$', r'$d$', r'$e$'])
#
# # gca = 'get current axis'
# ax = plt.gca()
# ax.spines['right'].set_color('none')
# ax.spines['top'].set_color('none')
# ax.xaxis.set_ticks_position('bottom')
# ax.yaxis.set_ticks_position('left')
# ax.spines['bottom'].set_position(('data', 0))
# ax.spines['left'].set_position(('data', 0))
#
# x0 = 0.4
# y0 = 2*x0 + 1
# plt.scatter(x0, y0, color = 'b')
# plt.plot([x0, x0], [y0, 0], 'k--', lw = 2.5)
# # method1
# plt.annotate(r'$2x+1=%s$' % y0, xy = (x0,y0), xycoords = 'data', xytext=(+30, -30), textcoords = 'offset points', fontsize = 16, arrowprops = dict(arrowstyle='->', connectionstyle='arc3, rad=0.2'))
#
# #method2
# plt.text(-1, 2, r'$This\ is\ the\ some\ text.\ \mu \sigma_i$')
#
# plt.show()

# 坐标轴数字透明
# x = np.linspace(-3, 3, 50)
# y = 0.2*x
#
# plt.figure(1)
# plt.plot(x, y, color='blue', linewidth=10.0, linestyle = '--')
# plt.xlim((-1, 2))
# plt.ylim((-2, 3))
# plt.xlabel('I am x')
# plt.ylabel('I am y')
# new_ticks = np.linspace(-1,2,5)
# print(new_ticks)
# # plt.xticks(new_ticks)
# # plt.yticks([0,-1.8,-3,1.22,3], [r'$\alpha$', r'$b$', r'$c$', r'$d$', r'$e$'])
#
# # gca = 'get current axis'
# ax = plt.gca()
# ax.spines['right'].set_color('none')
# ax.spines['top'].set_color('none')
# ax.xaxis.set_ticks_position('bottom')
# ax.yaxis.set_ticks_position('left')
# ax.spines['bottom'].set_position(('data', 0))
# ax.spines['left'].set_position(('data', 0))
#
# for label in ax.get_xticklabels() + ax.get_yticklabels():
#     label.set_fontsize(12)
#     label.set_bbox(dict(facecolor='white', edgecolor='None', alpha=0.7))
#
# plt.show()

# 散点图
# n = 1024
# X = np.random.normal(0, 1, n)
# Y = np.random.normal(0, 1, n)
# T = np.arctan2(Y, X)
# plt.scatter(X, Y, s = 75, c = T, alpha = 0.5)
# plt.xlim((-1.5, 1.5))
# plt.ylim((-1.5, 1.5))
# plt.xticks(())
# plt.yticks(())
# plt.show()

# 柱状图
# n = 12
# X = np.arange(n)
# Y1 = (1 - X/float(n))*np.random.uniform(0.5, 1.0, n)
# Y2 = (1 - X/float(n))*np.random.uniform(0.5, 1.0, n)
#
# plt.bar(X, +Y1, facecolor = '#9999ff', edgecolor = 'white')
# plt.bar(X, -Y2, facecolor = '#ff9999', edgecolor = 'white')
#
# for x, y in zip(X, Y1):
#     plt.text(x + 0.04, y + 0.05, '%.2f' % y, ha = 'center', va = 'bottom')
#
# for x, y in zip(X, Y2):
#     plt.text(x + 0.04, -y - 0.05, '%.2f' % y, ha = 'center', va = 'top')
#
# plt.xlim((-0.5, n))
# plt.ylim((-1.25, 1.25))
# plt.xticks(())
# plt.yticks(())
# plt.show()


# 等高线图
# def f(x, y):
#     return (1 - x/2 + x ** 5 + y ** 3) * np.exp(-x**2 - y**2)
#
# n = 256
# x = np.linspace(-3, 3, n)
# y = np.linspace(-3, 3, n)
#
# X, Y = np.meshgrid(x, y)
# plt.contourf(X, Y, f(X, Y), 8, alpha=0.75, cmap = plt.cm.hot)
#
# C = plt.contour(X, Y, f(X, Y), 8, colors = 'black', linewidth = 0.5)
#
# plt.clabel(C, inline = True, fontsize = 10)
#
# plt.xticks(())
# plt.yticks(())
# plt.show()

# 显示图像
# a = np.random.rand(3,3)
# plt.imshow(a, interpolation = 'nearest', cmap='bone', origin = 'lower')
# plt.colorbar(shrink = 0.9)
# plt.xticks(())
# plt.yticks(())
# plt.show()

# 显示 3D 图像
# fig = plt.figure()
# ax = Axes3D(fig)
# X = np.arange(-4, 4, 0.25)
# Y = np.arange(-4, 4, 0.25)
# X, Y = np.meshgrid(X, Y)
# R = np.sqrt(X**2 + Y**2)
# Z = np.sin(R)
#
# ax.plot_surface(X, Y, Z, rstride = 1, cstride = 1, cmap = plt.get_cmap('rainbow'))
# ax.contourf(X, Y, Z, zdir = 'x', offset = -4, cmap = 'rainbow')
#
# plt.show()

# 显示多个图 method1
# plt.figure(1)
#
# plt.subplot(2, 2, 1)
# plt.plot([0, 1], [0, 1])
# plt.subplot(2, 2, 2)
# plt.plot([0, 1], [0, 1])
# plt.subplot(2, 2, 3)
# plt.plot([0, 1], [0, 1])
# plt.subplot(2, 2, 4)
# plt.plot([0, 1], [0, 1])
#
# plt.figure(2)
# plt.subplot(2, 1, 1)
# plt.plot([0, 1], [0, 1])
# plt.subplot(2, 2, 3)
# plt.plot([0, 1], [0, 1])
# plt.subplot(2, 2, 4)
# plt.plot([0, 1], [0, 1])
#
#
# plt.show()

# 显示多个图 method2
import matplotlib.gridspec as gridspec
# plt.figure()
# ax1 = plt.subplot2grid((3,3), (0,0), colspan = 3, rowspan = 1)
# ax1.plot([0,3], [0,3])
# ax1.set_title('ax1_title')
#
# ax2 = plt.subplot2grid((3,3), (1, 0), colspan = 2)
# ax2.plot([0,3], [0,2])
#
# plt.tight_layout()
# plt.show()

# 显示多个图 method3
# gs = gridspec.GridSpec(3,3)
# ax1 = plt.subplot(gs[0,:])
# ax2 = plt.subplot(gs[1, :2])
# plt.show()

# 显示多个图 method4

# f, ((ax11, ax12), (ax21, ax22)) = plt.subplots(2,2, sharex = True, sharey = True)
# ax11.scatter([0,2], [1,1])
# plt.tight_layout()
# plt.show()


# 图中图
# fig = plt.figure()
# x = np.linspace(-1, 1, 20)
# y = x
# left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
# ax1 = fig.add_axes([left, bottom, width, height])
# ax1.plot(x,y,'r')
# ax1.set_xlabel('x')
# ax1.set_ylabel('y')
#
#
# left, bottom, width, height = 0.2, 0.6, 0.25, 0.25
# ax2 = fig.add_axes([left, bottom, width, height])
# ax2.plot(x,y,'r')
# ax2.set_xlabel('x')
# ax2.set_ylabel('y')
#
# plt.show()

# 主次坐标轴
# x = np.arange(0,10,0.1)
# y1 = 0.05*x**2
# y2 = -1*y1
# fig, ax1 = plt.subplots()
# ax2 = ax1.twinx()
# ax1.plot(x, y1, 'g-')
# ax2.plot(x, y2, 'b--')
# plt.show()

# 动画
from matplotlib import animation
fig, ax = plt.subplots()
x = np.arange(0, 2*np.pi, 0.01)
line, = ax.plot(x, np.sin(x))

def animate(i):
    line.set_ydata(np.sin(x + i/10))
    return line,

def init():
    line.set_ydata(np.sin(x))
    return line,

ani = animation.FuncAnimation(fig = fig, func = animate, frames = 100, init_func = init, interval=20, blit=True)

plt.show()