import matplotlib.pyplot as plt
import numpy as np

# x5 = [0.871,0.837,0.841]
# x8 = [0.842,0.799,0.807]
# x10 = [0.776,0.733,0.725]


# X = np.arange(3)


# fig,ax = plt.subplots(1)
# # print(help(ax.bar))
# ax.bar(X + 0.00,x5,color='b',width=0.25,tick_label=["precision","recall","f1score"], )
# ax.bar(X + 0.25, x8, color = 'g', width = 0.25,tick_label=["precision","recall","f1score"], )
# ax.bar(X + 0.50, x10, color = 'r', width = 0.25,tick_label=["precision","recall","f1score"], )
# ax.legend(["X5L","X8L","X10L"])
# fig.show()
# plt.show()


x5 = [0.871,0.837,0.841,0.91]
x8 = [0.842,0.799,0.807,0.895]
x10 = [0.776,0.733,0.725,0.861]


X = np.arange(4)


fig,ax = plt.subplots(1)
# print(help(ax.bar))
ax.bar(X + 0.00,x5,color='red',width=0.15, )
ax.bar(X + 0.25, x8, color = 'blue', width = 0.15, tick_label=["precision","recall","f1score","roc_auc"],)
ax.bar(X + 0.50, x10, color = 'green', width = 0.15, )
# ax.tick_label(["precision","recall","f1score","ROC AUC"],)

ax.legend(["X5L","X8L","X10L"])

fig.show()
plt.show()