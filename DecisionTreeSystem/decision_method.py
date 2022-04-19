# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 23:49:50 2022

@author: Team317
"""
from deal_data import cal_entropy, splitdataset
from math import log

def id3_choose_best_feature(dataset):
    '''
    ID3_chooseBestFeatureToSplit
    根据传入的数据集dataset，计算出用不同特征划分数据集得到的信息增益值
    其中的最大值对应的特征为最佳划分特征
    '''

    # 去掉一个特征之后的特征数
    numFeatures = len(dataset[0]) - 1

    # baseEnt是未划分前的信息熵
    baseEnt = cal_entropy(dataset)

    bestInfoGain = 0.0  # 记录最大信息增益
    bestFeature = 0     # 记录最大信息增益对应的特征

    # 由第i+1个特征来划分数据集
    for i in range(numFeatures):
        # 得到第i+1列样本数据
        featList = [example[i] for example in dataset]

        # 第i+1个特征的取值集合
        uniqueVals = set(featList)
        newEnt = 0.0    # 在第i+1个特征条件下，数据集的经验条件熵

        # 由标签列表中的取值划分数据集，计算不同取值的经验条件熵
        # H(Y|X)=sum( pi * H(Y|X=xi) )
        for value in uniqueVals:
            # 得到i+1个分量取值为value，且去掉第i+1个分量的样本集合
            subdataset = splitdataset(dataset, i, value)

            # 计算经验条件熵
            p = len(subdataset) / float(len(dataset))
            # 即 H(Y|X) = pi * H(Y|X=xi)
            newEnt += p * cal_entropy(subdataset)

        # 信息增益g(X,Y) = H(X) - H(Y|X)
        infoGain = baseEnt - newEnt

        # 保存信息增益最大的特征
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i

    # 返回信息增益最大的特征下标以及最大信息增益
    return bestFeature,bestInfoGain  


def c45_choose_best_feature(dataset):

    # 去掉一个特征之后的特征数
    numFeatures = len(dataset[0]) - 1

    # baseEnt是未划分前的信息熵
    baseEnt = cal_entropy(dataset)

    bestInfoGain_ratio = 0.0    # 记录最大信息增益率
    bestFeature = 0             # 记录最大信息增益率对应的特征

    # 由第i+1个特征来划分数据集
    for i in range(numFeatures):
        # 第i+1列样本数据
        featList = [example[i] for example in dataset]
        # 第i+1个特征的取值集合
        uniqueVals = set(featList) 
        empiricalConditionalEntropy = 0.0   # 经验条件熵
        empiricalEntropy = 0.0              # 经验熵
        for value in uniqueVals:  # 计算每种划分方式的信息熵
            # 得到i+1个分量取值为value，且去掉第i+1个分量的样本集合
            subdataset = splitdataset(dataset, i, value)

            # 计算经验条件熵
            p = len(subdataset) / float(len(dataset))
            # 即 H(Y|X) = sum( pi * H(Y|X=xi) )
            empiricalConditionalEntropy += p * cal_entropy(subdataset)
            # 经验熵 即 IV(A) = - sum( p * log_2(p) )
            empiricalEntropy = empiricalEntropy - p * log(p, 2)

        # 信息增益
        infoGain = baseEnt - empiricalConditionalEntropy
        # 经验条件熵为0则不进行计算
        if (empiricalEntropy == 0):
            continue
        # 信息增益率
        infoGain_ratio = infoGain / empiricalEntropy

        # 保存信息增益率最大的特征
        if (infoGain_ratio > bestInfoGain_ratio):
            bestInfoGain_ratio = infoGain_ratio
            bestFeature = i

    # 返回信息增益最大的特征下标以及最大信息增益
    return bestFeature, bestInfoGain_ratio


def cart_choose_best_feature(dataset):

    # 去掉一个特征之后的特征数
    numFeatures = len(dataset[0]) - 1

    bestGini = 999999.0   # 记录最小基尼指数
    bestFeature = 1       # 记录最大基尼指数对应的特征
    # 由第i个特征来划分数据集
    for i in range(numFeatures):
        # 得到第i+1列样本数据
        featList = [example[i] for example in dataset]

        # 由特征列表的取值得到取值集合，形成无重复元素的标签列表
        uniqueVals = set(featList)

        # 由标签列表中的取值划分数据集，计算不同取值的基尼指数
        # 对于给定的特征A，若当A=ai时Gini(D,A=ai)最大，
        # 则ai为A的最优切分点，同时Gini(D,A=ai)做为A的基尼指数
        # Gini(D) = 1 - sum( p^2 ) p=|di|/|D|;对于二分类问题，Gini(D) = 2p(1-p)
        # Gini(D, A=ai) = |ai|/|D|*Gini(A=ai) + |!ai|/|D|*Gini(!ai)
        # 注:|D|表示聚合D的样本数量;|ai|表示A=ai的样本数量;|di|表示分类结果为di的样本数量
        for value in uniqueVals:
            # 在第i+1个特征条件下，且该特征取值为value时，该特征的基尼指数
            gini = 0
            # 将数据集依据第i+1个特征的取值是否为value划分为两个数据集
            restDataset, anotherDataset = splitdataset(dataset, i, value, returnAnother=True)
            # 如果的数据集中无样本，则不计算基尼指数
            if len(restDataset) == 0 or len(anotherDataset) == 0:
                continue

            # 数据集restDataset的基尼指数
            subp = len(splitdataset(restDataset, -1, '1')) / float(len(restDataset))
            gini1 = (1.0 - pow(subp, 2) - pow(1 - subp, 2))
            # 数据集anotherDataset的基尼指数
            subp = len(splitdataset(anotherDataset, -1, '-1')) / float(len(anotherDataset))
            gini2 = (1.0 - pow(subp, 2) - pow(1 - subp, 2))

            # 第i+1个特征取值为value的频率
            p = len(restDataset) / float(len(dataset))
            # 第i+1个特征取值为value时的基尼指数
            gini = p*gini1 + (1-p)*gini2

            # 记录最小的基尼指数及对应的特征下标
            if (gini < bestGini):
                bestGini = gini
                bestFeature = i

    return bestFeature, bestGini



