#!/usr/bin/env python3

import DO_ML
from DO_ML.do_common_libraries import *
# from matplotlib import rcParams
from DO_ML.do_test import *

# rcParams["figure.figsize"] = 5, 4
# sns.set(font_scale=1.5)


def corr_plot(df, size=10, **kwargs):
    '''size: vertical and horizontal size of the plot'''
    corr = df.corr()
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    fig, ax = plt.subplots(figsize=(size, size))
    cmap = sns.diverging_palette(200, 10, as_cmap=True)
    cplot = sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0, fmt=".2f", ax=ax, annot=True, annot_kws={"size":12}, square=True, linewidths=0.25, cbar=True, cbar_kws={"shrink":0.6, "extend":"both", "ticks":[-1.0,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1.0], "label":"correlation"}, **kwargs)
    cplot.set_xticklabels(cplot.get_xticklabels(), rotation=45,fontsize="medium",horizontalalignment='right', fontweight='light')
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)

    # ax.legend()
    # cax = ax.matshow(corr)
    # fig.colorbar(cax)
    # plt.xticks(range(len(corr.columns)), corr.columns, rotation="vertical")
    # plt.yticks(range(len(corr.columns)), corr.columns)


def regression_plot(xcolumn_name, ycolumn_name, df, scatter):
    sns.regplot(x=xcolumn_name, y=ycolumn_name, data=df, scatter=scatter)


def count_plot(xcolumn_name, df, palette="hls"):
    sns.countplot(x=xcolumn_name, data=df, palette=palette)


def pair_plot(df, column=None):  # scatter plot matrix
    sns.pairplot(data=df, hue=column, palette="hls")


def heatmap_plot(df):
    sns.heatmap(data=df, xticklabels=df.columns.values, yticklabels=df.columns.values)


def box_plot(x, y, df, palette="hls"):
    sns.boxplot(x=x, y=y, data=df, palette=palette)


def distribution_plot(column):  # bar plot
    sns.distplot(column)


def graph_plot(xcolumn, ycolumn, xlabel, ylabel, title, size):
    """ size: size of xticks"""
    fig = plt.figure()
    ax = fig.add_axes([.1, .1, 1, 1])
    ycolumn.plot()
    ax.set_xticks(range(size))
    ax.set_xticklabels(xcolumn, rotation=60, fontsize="medium")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="best")
    ax.set_ylim([ycolumn.min(), ycolumn.max()])
    ax.annotate("Label", xy=(0,0), xytext=(1,1), arrowprops=dict(facecolor="black", shrink=0.05))






def main():
    pass

if __name__ == '__main__': main()