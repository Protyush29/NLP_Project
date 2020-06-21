import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


class Plot:
    def plot_ROC(self,x,y,ROC):
        plt.plot(x, y, label='area = %.2f' % ROC)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.legend(loc='lower right')
        plt.show()
