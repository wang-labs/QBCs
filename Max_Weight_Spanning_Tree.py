"""
***************************You can read me*************************
# Authors:Xiao-Ying Zhang  & Ming-Ming Wang
# create_data:2024/5/29
*******************************************************************
"""
import numpy as np
import copy

Weight_Table_Mnist = np.array([[0,9.65243454,11.5119077,10.63770291,13.7229214,8.91505616,11.79632209,11.41713319,13.8073]
    ,[0,0,8.78281168,7.287634,9.34752118,8.93720541,9.1121826,7.78204956,9.46785]
    ,[0,0,0,8.94131955,10.937259,8.20466138,11.10420681,9.79942478,11.13824397]
    ,[0,0,0,0,10.48490065,6.89267772,9.52952485,10.93946532,10.40690538]
    ,[0,0,0,0,0,8.76455542,11.17762778,11.03543305,13.68221551]
    ,[0,0,0,0,0,0,8.83736018,7.21638853,8.86279083]
    ,[0,0,0,0,0,0,0,10.26702749,11.33017882]
    ,[0,0,0,0,0,0,0,0,11.24464274]
    ,[0,0,0,0,0,0,0,0,0]])

Weight_Table_FashionMnist = np.array([[0,13.2037971,17.17762687,16.99615927,19.02431302,16.53576114,17.303167,16.83518872,20.17733701]
    ,[0,0,13.44009266,12.72947663,13.54689799,12.27023235,12.61168763,13.82328949,13.15893659]
    ,[0,0,0,16.9110484,15.68886636,15.76264448,16.83505643,15.81046944,16.41753914]
    ,[0,0,0,0,16.31710288,17.95455869,16.30197591,16.41259237,18.55998001]
    ,[0,0,0,0,0,15.83355091,15.83573967,16.83174376,18.29406616]
    ,[0,0,0,0,0,0,17.1529929,16.47115484,18.01216779]
    ,[0,0,0,0,0,0,0,17.0735397,16.5204509]
    ,[0,0,0,0,0,0,0,0,17.04969647]
    ,[0,0,0,0,0,0,0,0,0]])


def search_connect_component(Tree,start,end,previous=None):
    """
    Args:
        Tree:
        start:
        end:
        previous:
    Returns:
        True,False
    """
#
    if start >end:
        start,end = end,start
    for next_node,is_available in enumerate(Tree[start,:]):
        if is_available != 0:
            if next_node == end:
                return True
            else:
                if previous is not None:
                    if next_node == previous:
                        continue
                if search_connect_component(Tree,next_node,end,start):
                    return True
                else:
                    continue
    return False

def Kruskal(weight_table:np.ndarray):
    item = 1
    Tree = np.zeros(shape=weight_table.shape)
    number_of_cloumn = weight_table.shape[1]
    while Weight_Table_Mnist.any() != 0:
        max_wight_index = np.argmax(weight_table)
        row, col = np.divmod(max_wight_index, number_of_cloumn)
        if not search_connect_component(Tree, row, col):
            Tree[row, col] = 1
            Tree[col, row] = 1
        weight_table[row,col] = 0
        item += 1
        if item <40:
            pass
        else:
            break
    return Tree

def trans_to_tree(graphy):
    graphy_local = copy.deepcopy(graphy)
    tree_list = [[],[],[],[],[],[],[],[],[]]
    parents_node = []
    nodes = []
    while len(nodes)<graphy_local.shape[0]:

        root = np.argmax(np.sum(graphy_local,axis=1))
        if root not in nodes:
            nodes.append(root)
            parents_node.append(root)

        for node,edge in enumerate(graphy_local[root,:]):
            if edge!=0:
                graphy_local[root, node] = -1
                if node not in nodes:
                    tree_list[root].append(node)
                if node in nodes:
                    if root in tree_list[node]:
                        continue
                    else:
                        tree_list[node].append(root)
                nodes.append(node)
    return tree_list

def deep_of_node(tree_list):
    deep_list = np.zeros(shape=(9,))
    for node,childrens in enumerate(tree_list):
        for children in childrens:
            deep_list[children] = deep_list[node] +1
    return deep_list

def tans_to_parent_tree(tree_list):
    parent_tree = [[],[],[],[],[],[],[],[],[]]
    for parent,childrens in enumerate(tree_list):
        for children in childrens:
            parent_tree[children].append(parent)
    return parent_tree

def get_tree_structure(set):
    if set == "mnist":
        graphy = Kruskal(Weight_Table_Mnist)
        tree_list = trans_to_tree(graphy)
        return tree_list
    elif set == "fashion_mnist":
        graphy = Kruskal(Weight_Table_FashionMnist)
        tree_list = trans_to_tree(graphy)
        return tree_list
    else:
        raise Exception("not data")


