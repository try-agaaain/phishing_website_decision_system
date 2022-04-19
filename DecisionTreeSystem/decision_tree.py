# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 23:43:18 2022

@author: Team317
"""

from classify import classify_test
from deal_data import cal_acc, majorityCnt, splitdataset
from collections import Counter
PRE_CUT = False     # 预剪枝
POST_CUT = True     # 后剪枝
NODE_MAEK = 0       # 节点标号


def create_tree(dataset, labels, test_dataset, chooseBestFeature):
    '''
    递归的创建决策树
    '''

    # 节点编号，是全局变量，附加于节点名称中以区分不同的节点，方便pydot绘制
    global NODE_MAEK    # 在第六行进行了定义，前面加上global进行全局变量声明
    global POST_CUT, PRE_CUT
    # 调试点，决策树是用深度优先遍历的方式来创建的，
    # 通过判断NODE_MAEK可调试决策树中特定节点的生成过程
    # break_point = 7
    # if NODE_MAEK == break_point:
    #     print("This is {}th node.".format(break_point))

    # 最后一列是判别结果
    classList = [example[-1] for example in dataset]

    # 当dataset中只包含判别结果而不包含特征时，
    # 将占比最大的取值作为结果返回，成为树的叶节点
    if len(dataset[0]) == 1:
        samples,classify = majorityCnt(classList)
        leafInfo = "Node ID={mark}\nSamples={samples}\nClass={classify}".format(
            classify = classList[0],
            samples = len(classList),
            mark = NODE_MAEK)
        return leafInfo

    # 如果最后一列的取值全部相同，将该取值作为结果返回，成为叶节点
    if classList.count(classList[0]) == len(classList):
        leafInfo = "Node ID={mark}\nSamples={samples}\nClass={classify}".format(
            mark = NODE_MAEK,
            samples=len(classList),
            classify=classList[0])
        return leafInfo

    # 得到信息增益最大的特征下标
    bestFeat, bestInfoGain = chooseBestFeature(dataset)
    # 信息增益最大的特征对应的名称
    bestFeatLabel = labels[bestFeat]

    # 为信息增益最大的特征创建树节点（非叶节点）
    sample_counter = Counter()
    for vec in dataset:
        sample_counter[vec[-1]] += 1

    bestFeatInfo = "{featureName}\nNode ID={mark}\nGini/Gain={gini:.5f}\nSamples={samples}".format(
                    featureName=bestFeatLabel,
                    mark = NODE_MAEK,
                    gini = bestInfoGain,
                    samples = dict_to_str(sample_counter))
    decisionTree = {bestFeatInfo: {}}

    # 保存副本，后剪枝时featLabels作为参数传递
    featLabels = labels[:]
    # 从特征名称列表中去掉当前信息增益最大的特征
    del (labels[bestFeat])

    # 得到数据集中信息增益最大的特征的取值集合
    featValues = [example[bestFeat] for example in dataset]
    uniqueVals = set(featValues)

    # 进行预剪枝
    if PRE_CUT:
        #----------计算不进行划分（即返回叶节点）的情况下，该子树决策的准确率----------#
        # ans记录测试样本的实际分类结果，也就是test_dataset的最后一列
        ans = []
        for index in range(len(test_dataset)):
            ans.append(test_dataset[index][-1])

        # result_count记录不同分类结果的样本数量
        result_counter = Counter()
        for vec in dataset:
            result_counter[vec[-1]] += 1

        # leaf_output得到样本数量最多的分类取值
        cut_output = result_counter.most_common(1)[0][0]
        # 进行预剪枝，采用多数投票来分类，计算此时的准确率
        cut_acc = cal_acc(test_output=[cut_output] * len(test_dataset), label=ans)

        #----------计算进行划分的情况下，该子树决策在测试集上的准确率----------#
        outputs = []    # 划分子集后采用多数投票法对测试集进行预测的结果
        ans = []        # 测试集样本的实际分类结果
        # 对于当前特征的不同取值来划分数据集，使用多数投票法来预测各子集的决策结果
        for value in uniqueVals:
            # 保留测试集中取值为value、信息增益最大的特征样本，再去掉该特征对应那一列的数据
            split_testset = splitdataset(test_dataset, bestFeat, value)
            # 提取测试集中的分类结果
            for vec in split_testset:
                ans.append(vec[-1])

            # 保留训练集中取值为value、信息增益最大的特征样本，再去掉该特征对应那一列的数据
            split_dataset = splitdataset(dataset, bestFeat, value)
            # 统计训练集上不同分类结果的样本数
            result_counter = Counter()
            for vec in split_dataset:
                result_counter[vec[-1]] += 1
            # 计算训练集中占比最大的分类结果取值
            leaf_output = result_counter.most_common(1)[0][0]
            # 用多数投票法对测试集进行预测
            outputs += [leaf_output] * len(split_testset)

        # 计算验证集准确率
        uncut_acc = cal_acc(test_output=outputs, label=ans)

        # 比较划分前后测试集上的准确率，以决定当前节点是否作为叶子节点
        if cut_acc >= uncut_acc:
            leafInfo = "Node ID={mark}\nSamples={samples}\nClass={classify}".format(
                            mark = NODE_MAEK,
                            samples = dict_to_str(sample_counter),
                            classify = cut_output)
            return leafInfo



    # 当前节点为非叶子节点，则需进一步构建子树
    for value in uniqueVals:

        NODE_MAEK += 1

        # 浅复制
        subLabels = labels[:]  
        # 数据集和测试集中去除当前特征分量，进一步构建决策树
        decisionTree[bestFeatInfo][value] = create_tree(
            splitdataset(dataset, bestFeat, value),
            subLabels,
            splitdataset(test_dataset, bestFeat, value),
            chooseBestFeature)


    # 后剪枝，如果划分后测试子集test_dataset中没有样本，则此时无法通过测试集来判断剪枝效果，故不剪枝
    if POST_CUT and len(test_dataset) != 0 :

        # 测试集的预测结果
        tree_output = classify_test(decisionTree,
                                   featLabels=featLabels,
                                   testDataSet=test_dataset)
        # 测试样本中的实际分类结果
        ans = []
        for vec in test_dataset:
            ans.append(vec[-1])
        # 计算不剪枝的准确率
        uncut_acc = cal_acc(tree_output, ans)

        # 统计不同分类结果的样本数量
        result_counter = Counter()
        for vec in dataset:
            result_counter[vec[-1]] += 1
        # 采用多数表决进行预测
        leaf_output = result_counter.most_common(1)[0][0]
        # 多数表决预测下的准确率，即剪枝后的准确率
        cut_acc = cal_acc([leaf_output] * len(test_dataset), ans)

        # 比较剪枝前后的准确率
        if cut_acc >= uncut_acc:
            leafInfo = "Node ID={mark}\nSamples={samples}\nClass={classify}".format(
                            mark = NODE_MAEK,
                            samples = dict_to_str(sample_counter),
                            classify = leaf_output)
            return leafInfo

    # 不需要预剪枝，也不需要进行后剪枝，返回生成的子树
    return decisionTree

def dict_to_str(a_dict):
    dict_info = str(dict(a_dict)).split(":")
    return '/'.join(dict_info)


def change_config(cut):
    '''
    全局变量PRE_CUT, POST_CUT属于decision_tree，在外部修改后只在外部生效，
    返回到decision_tree后PRE_CUT, POST_CUT仍然保持原样，
    所以如果想在外部修改decision_tree中的全局变量，还得借助decision_tree中定义的函数
    '''
    global PRE_CUT, POST_CUT
    PRE_CUT, POST_CUT = cut