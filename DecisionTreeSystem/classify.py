# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 10:19:38 2022

@author: Team317
"""


def classify(inputTree, featLabels, testVec):
    """
    输入决策树、分类标签和测试样本，根据已得到的决策树递归的进行预测
    注意testVec是一个样本，而非样本集
    """
    # 当前节点的名称
    firstInfo = list(inputTree.keys())[0]
    firstStr = firstInfo.split('\n')[0]

    # 当前节点的孩子节点
    secondDict = inputTree[firstInfo]
    # 当前节点的特征名称在列表中的下标
    featIndex = featLabels.index(firstStr)

    # 默认归类为'0'（'1'表示钓鱼网站，'-1'表示非钓鱼网站，'0'可以表示未分类）
    classLabel = '0'
    for key in secondDict.keys():
        # 样本的当前特征取值与key一致，则沿决策树的分支进行决策
        if testVec[featIndex] == key:
            # 如果是字典类型，则表示该分支为子树，需顺着决策树确定分类
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)

            # 否则为叶子节点，得到决策分类，这里可以直接返回
            else:
                classLabel = secondDict[key].split('=')[-1]
    return classLabel


def classify_test(inputTree, featLabels, testDataSet):
    """
    输入决策树、分类标签和测试数据集，对测试数据集进行预测
    """
    classLabelAll = []
    for testVec in testDataSet:
        classLabelAll.append(classify(inputTree, featLabels, testVec))
    return classLabelAll
