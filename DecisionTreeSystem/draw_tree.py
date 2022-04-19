# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 13:34:22 2022

@author: Team317
"""

from random import sample
import pydot
class DrawTree:
    graph = None

    def __init__(self, brief=False):
        '''
        使用graphiz绘制决策树
          - brief: 为True表示是否采用简要方式绘制
        '''
        # 创建图
        self.graph = pydot.Dot(graph_type='graph')
        # 记录边数
        self.edgeNum = 0
        self.brief = brief  # 是否采用简要方式绘制
        # 节点颜色集合
        self.COLOR = {  "BLUE": "#399de5",
                        "LIGHT_BLUE": "#88c4ef",
                        "WHITE": "#FFFFFF",
                        "LIGHT_YELLOW": "#eeae80",
                        "YELLOW": "#e78d4b",
                    }

    def draw(self, parent_name, child_name, label):
        # 生成子节点，并设置相关的参数
        shape = "box"
        if(child_name.find("Class") != -1):
            shape = "ellipse"
        color = self.get_color(child_name)
        child_name = self.get_name(child_name)
        parent_name = self.get_name(parent_name)
        node = pydot.Node(name=child_name, shape=shape, style="filled, rounded",
                        color="black", fillcolor=color, fontname="helvetica")
        edge = pydot.Edge(parent_name, child_name, label=label)
        # 将生成的边和节点加入到图中
        self.graph.add_node(node)
        self.graph.add_edge(edge)
        self.edgeNum += 1

    def visit(self, node, parent=None, label=None):  # 运用了递归
        # 根节点 parent为None，则node为根节点
        if parent == None:
            parent = list(node.keys())[0]
            children = node[parent]
            # 将根节点加入树中
            color = self.get_color(parent)
            name = self.get_name(parent)
            node = pydot.Node(name=name, shape='box', style="filled, rounded",
                            color="black", fillcolor=color, fontname="helvetica")
            self.graph.add_node(node)
            # 绘制子树
            self.visit(children, parent)
            return

        # 叶节点 node为str类型，则node为叶节点
        if isinstance(node, str):
            self.draw(parent, node, label)
            return

        # 中间节点
        for key, value in node.items():

            # 当key取-1,1,0时，表示决策边，而不是节点
            if key in {'-1', '1', '0'}:
                self.visit(value, parent, key)

            # 否则，表示的是节点，需要添加边
            else:
                # 添加边，label为决策父节点的一个取值
                self.draw(parent, key, label)
                # 如果该节点不是叶子节点，则需遍历子树
                if isinstance(value, dict):
                    self.visit(value, key, label)

    def start(self, dic, filename):
        # 生成决策树
        self.visit(dic)
        # 生成PNG
        self.graph.write_png(
            './Images/{}.png'.format(filename))  # 最后结果生成png图片格式

    def get_nodeNum(self):
        '''
        返回树中的节点的数量
        树的节点数 = 边数 + 1
        '''
        return self.edgeNum + 1

    def get_color(self, node_info):

        info = node_info.split("=")[-1]

        # 如果info为1或-1，则表示该节点为叶子节点，由节点分类返回颜色
        if info == '-1':
            return self.COLOR["BLUE"]
        if info == '1':
            return self.COLOR["YELLOW"]
        # 否则该节点为中间节点，info是一个样本计数字典，由样本比例来确定颜色
        info_list = info.split('/')
        sample_info = ':'.join(info_list)
        sample_dict = eval(sample_info)

        try:
            if sample_dict['-1'] > sample_dict['1']:
                return self.COLOR['LIGHT_BLUE']
            elif sample_dict['-1'] < sample_dict['1']:
                return self.COLOR['LIGHT_YELLOW']
            else:
                return self.COLOR['WHITE']
        except:
            pass

    def get_name(self, node_info):
        if self.brief:
            info_list = node_info.split('\n')
            node_name = info_list[0] + '\n' + info_list[1]
            return node_name
        return node_info