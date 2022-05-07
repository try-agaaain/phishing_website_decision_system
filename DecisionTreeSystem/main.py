import cv2
from decision_method import id3_choose_best_feature, \
                            c45_choose_best_feature, \
                            cart_choose_best_feature
from draw_tree import DrawTree
from decision_tree import create_tree, change_config
from classify import classify_test
from deal_data import cal_acc, read_datas


def summary_result(labels, data_train, data_test):
    '''
    将两种剪枝算法和三种特征选择算法相结合运用于数据集，
    构建生成树，将结果统计为表格，其中包含准确率和树的节点数
    '''
    # 四种剪枝设置
    cut_configs = (False, False), (True, False), (False, True), (True, True)
    # 三种决策树特征选择算法
    methods = {'ID3': id3_choose_best_feature,
               'C45': c45_choose_best_feature,
               'CART': cart_choose_best_feature}

    # 记录准确率最大的决策树
    bestTree = None  # 最佳的特征选择算法
    maxAccuracy = 0  # 准确率
    prune = None  # 预剪枝和后剪枝设置
    # ----------绘制各种算法和剪枝方法结果比对表----------#
    # 打印表头
    print('预剪枝,后剪枝   ID3/节点数   C45/节点数    CART/节点数')
    # data_temp = data_test # 备份测试集
    for cut in cut_configs:

        # 设置剪枝策略
        change_config(cut)

        # 打印剪枝设置
        cut_info = '{:^5}_{:^5}'.format(str(cut[0]), str(cut[1]))
        print(cut_info + '\t', end='')

        for methodName, method in methods.items():
            # 生成决策树
            labels_tmp = labels[:]  # 拷贝，createTree会改变labels

            # data_test = data_temp   # 将data_test改回为测试集
            decisionTree = create_tree(
                data_train,
                labels_tmp,
                data_test,
                method)
            # data_test = data_train  # 如果要计算训练集上的准确率，则在计算前将测试集改为训练集
            # 对测试集进行预测
            pre_result = classify_test(decisionTree, labels, data_test)
            # 样本的实际分类结果
            actual_result = [sample[30] for sample in data_test]
            # 计算准确率
            accuracy = cal_acc(pre_result, actual_result)

            # 绘制决策树
            photo_draw = DrawTree(brief=False)
            # 生成文件名
            filename = methodName + '_' + cut_info
            photo_draw.start(decisionTree, filename)
            # 获得决策树的节点数
            nodeNum = photo_draw.get_nodeNum()
            # 打印准确率
            print(f'{accuracy:.4f}/{nodeNum:<6d}', end='')
            # 记录各种算法中最佳的决策树
            if accuracy > maxAccuracy:
                bestTree = methodName
                maxAccuracy = accuracy
                prune = cut
            # 展示据册数
            img = cv2.imread(f'../Images/{filename}.png')
            img = cv2.resize(img, (2560, 1440))
            cv2.imshow(filename, img)
            cv2.waitKey(0)
        print()  # 打印换行符

    # 打印最佳决策树结果
    print(f'最好的决策方法为{bestTree}\n',
          f'准确率为{maxAccuracy}\n'
          f'剪枝情况为 预剪枝：{prune[0]} 后剪枝：{prune[1]}')


def front_n_layer_tree(tree, current, target):
    '''
    进行后序遍历砍掉第n层下面的子树，
    从而获取嵌套字典（也即决策树）的前n层
    '''
    if current == target:
        # 当达到指定层时返回空节点
        return None

    # 当前节点为叶子节点，则直接返回
    if isinstance(tree, str):
        return tree

    for parent, child in tree.items():
        node = None
        # 子节点是非叶节点
        # 如果当前节点为边，则跳过本层，直接进入下一层
        if parent in {'-1', '1', '0'}:
            node = front_n_layer_tree(child, current, target)

        elif isinstance(child, dict):
            node = front_n_layer_tree(child, current + 1, target)

        tree[parent] = node

    return tree


def draw_front_tree(labels, data_train, data_test, method, front=None):
    # 拷贝标签
    labels_tmp = labels[:]
    decisionTree = create_tree(
        data_train,
        labels_tmp,
        data_test,
        method)
    if front != None:
        subtree = front_n_layer_tree(decisionTree, 0, front)
        filename = f'id3决策树前{front}层'
    else:
        subtree = decisionTree
        filename = "决策树系统"
    # 绘制决前front层子树
    photo_draw = DrawTree(brief=False)

    photo_draw.start(subtree, filename)


if __name__ == '__main__':
    # 全部特征以及判别结果的名称
    labels = ["having_IP_Address", "URL_Length", "Shortining_Service",
              "having_At_Symbol", "double_slash_redirecting",
              "Prefix_Suffix", "having_Sub_Domain", "SSLfinal_State",
              "Domain_registeration_length", "Favicon", "port",
              "HTTPS_token", "Request_URL", "URL_of_Anchor",
              "Links_in_tags", "SFH", "Submitting_to_email",
              "Abnormal_URL", "Redirect", "on_mouseover", "RightClick",
              "popUpWidnow", "Iframe", "age_of_domain", "DNSRecord",
              "web_traffic", "Page_Rank", "Google_Index",
              "Links_pointing_to_page", "Statistical_report", "Result"]

    # 读取数据并划分为训练集和测试集
    filename = '../Datas/UniqueDatas.txt'
    data_train, data_validation, data_test = read_datas(filename)

    summary_result(labels, data_train, data_validation)  # 查看整体结果

    # 绘制前三层决策树
    labels_tmp = labels[:]
    change_config((False, True))
    draw_front_tree(labels_tmp, data_train, data_test, id3_choose_best_feature, 10)
