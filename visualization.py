
import matplotlib.pyplot as plt

class visualization:

    def __init__(self):
        config = {
            "font.family":"Arial",
            "mathtext.fontset":'cm',
            "font.size":15,
            "axes.linewidth":2.5,
            "xtick.labelsize":15,
            "ytick.labelsize":15,
            "xtick.direction":'in',
            "ytick.direction":'in',
            "xtick.top":True,
            "ytick.right":True,
            "xtick.major.width":3,
            "xtick.major.size":8,
            "xtick.minor.visible":True,
            "xtick.minor.width":3,
            "xtick.minor.size":5,
            "ytick.major.width":3,
            "ytick.major.size":8,
            "ytick.minor.visible":True,
            "ytick.minor.width":3,
            "ytick.minor.size":5,
            "figure.figsize":[8,4],
            "pdf.fonttype":42,
            "ps.fonttype":42
        }

        plt.rcParams.update(config)
        super().__init__()

    def plot_yerr(self,x,y,yerr,xlabel,ylabel,label,fmt='o',color='k',legend=True):
        plt.errorbar(x,y, yerr =yerr,capsize=5, fmt=fmt, \
            markersize=14, ecolor=color, markeredgecolor = color, color='w',\
                barsabove=True,markeredgewidth=2,label=label)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xlim(min(x)*(1-0.1),max(x)*(1+0.1))
        plt.ylim(min(y)*(1-0.1),max(y)*(1+0.1))
        if legend:
            plt.legend(loc='best',numpoints=1).get_frame().set_alpha(0)
        return plt