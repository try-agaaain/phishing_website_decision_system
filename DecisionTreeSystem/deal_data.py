import operator
import random

from sklearn.model_selection import train_test_split
from math import log


def read_datas(filename, train_ratio, valitation_ratio, test_ratio):
    """
    读取全部数据，返回训练集、验证集、测试集、标签，
    训练集：验证集：测试集的比例为 train_ratio:valitation_ratio:test_ratio
    """

    fr = open(filename, 'r')
    all_lines = fr.readlines()  # list形式,每行为1个str

    # 用列表保存全部数据，数据元素的类型为str
    dataset = []
    for line in all_lines[0:]:
        line = line.strip().split(',')  # 以逗号为分割符拆分列表
        dataset.append(line)

    # random.shuffle(dataset)
    # 使用sklearn库来划分数据集
    # data_train, data_test = train_test_split(
    #         dataset, test_size=0.33, random_state=42, shuffle=True)
    train_number = int(len(dataset) * train_ratio)
    valitation_number = int(len(dataset) * valitation_ratio)
    split_point1, split_point2 = train_number, train_number+valitation_number
    data_train, data_validation, data_test = dataset[:split_point1], \
                                         dataset[split_point1:split_point2 * 2], \
                                         dataset[split_point2:]

    return data_train, data_validation, data_test


def splitdataset(dataset, axis, value, returnAnother=False):
    '''
    划分数据集，保留第axis+1列取值为value的样本，并去掉第axis列
    '''

    restDataset = []  # 记录第axis+1列取值为value的样本，并去掉第axis列
    anotherDataset = []  # 记录不符合条件的样本集
    # 抽取符合划分特征的值
    for featVec in dataset:

        # 去掉样本的第axis+1个分量
        reducedfeatVec = featVec[:axis]
        reducedfeatVec.extend(featVec[axis + 1:])

        # 如果样本的第axis+1列的值为value，则保存到restDataset中
        if featVec[axis] == value:
            restDataset.append(reducedfeatVec)

        # 否则，保存到anotherDataset中
        else:
            anotherDataset.append(reducedfeatVec)
    if returnAnother:
        return restDataset, anotherDataset

    return restDataset


def majorityCnt(classList):
    '''
    classList是判别结果，当数据集中只包含判别结果时，
    使用多数表决来决定分类结果
    '''
    classCont = {}
    # 统计各个判别结果的样本数，取其中样本数最大的判别结果作为叶节点返回
    for vote in classList:
        if vote not in classCont.keys():
            classCont[vote] = 0
        classCont[vote] += 1
    sortedClassCont = sorted(classCont.items(),
                             key=operator.itemgetter(1),
                             reverse=True)
    return dict(sortedClassCont), sortedClassCont[0][0]


def cal_acc(test_output, label):
    """
    计算准确率，输入预测结果列表test_output和实际结果列表label
    """
    # 断言，判断test_output和label的长度是否相等
    assert len(test_output) == len(label)
    # count统计预测正确的样本数量
    count = 0
    for index in range(len(test_output)):
        if test_output[index] == label[index]:
            count += 1

    # 返回正确率
    return float(count / len(test_output))


def cal_entropy(dataset):
    '''
    计算信息熵 entropy = -sum( pi*log(pi) )
    '''
    try:
        # 样本数量
        numEntries = len(dataset)

        # 给所有可能分类创建字典
        labelCounts = {}

        # 统计最后一个特征各分类的频数，记录在labelCounts中
        for featVec in dataset:
            currentlabel = featVec[-1]
            # 发现新类别，则加入字典并将其频数设置为0
            if currentlabel not in labelCounts.keys():
                labelCounts[currentlabel] = 0
            labelCounts[currentlabel] += 1
    except:
        print()

    # 计算信息熵
    Ent = 0.0
    for key in labelCounts:
        p = float(labelCounts[key]) / numEntries
        Ent = Ent - p * log(p, 2)  # 以2为底求对数

    return Ent
