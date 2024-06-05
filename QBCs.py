"""
***************************You can read me*************************
# Authors:Xiao-Ying Zhang  & Ming-Ming Wang
# create_data:2024/5/29
*******************************************************************
"""
from typing import Union

import numpy as np
from mindquantum.core import Circuit,X,RY
from mindquantum.simulator import Simulator
from Max_Weight_Spanning_Tree import *
import pandas as pd
class QBCs():
    def __init__(self,set,label_list,structure,radius=3,distance = None):
        self.set = set
        self._label_list = label_list
        self._structure = structure
        self._circuit = self.Bayesian_circuit(self._structure)
        self._radius = radius
        self._distance = distance
        self._probability_table = None
        self._intersections = None
        self._targets =None
        self._de_relabel_dic =None
        self._en_relabel_dic = None

    def Get_dataset(self,set: str = 'mnist',label_list:list = None):
        if set == 'mnist':
            dataset = np.load("mnist.npz", allow_pickle=True)
            x_train = dataset['x_train']
            y_train = dataset['y_train']
            x_test = dataset['x_test']
            y_test = dataset['y_test']
        elif set == 'fashion_mnist':
            x_train = np.load("./fashion_mnist/train_x.npy", allow_pickle=True)
            y_train = np.load("./fashion_mnist/train_y.npy", allow_pickle=True)
            x_test = np.load("./fashion_mnist/test_x.npy", allow_pickle=True)
            y_test = np.load("./fashion_mnist/test_y.npy", allow_pickle=True)
        if len(label_list) < 2:
            raise Exception("Input two labels")
        else:
            new_x_train = np.ndarray(shape=(0, 28, 28))
            new_y_train = np.ndarray(shape=(0))
            new_x_test = np.ndarray(shape=(0, 28, 28))
            new_y_test = np.ndarray(shape=(0))
            for cls in label_list:
                new_x_train = np.concatenate([new_x_train, x_train[y_train == int(cls), :, :]], axis=0)
                new_y_train = np.concatenate([new_y_train, y_train[y_train == int(cls)]], axis=0)
                new_x_test = np.concatenate([new_x_test, x_test[y_test == int(cls), :, :]], axis=0)
                new_y_test = np.concatenate([new_y_test, y_test[y_test == int(cls)]], axis=0)
            return new_x_train, new_y_train, new_x_test, new_y_test

    # real labels to 0,1
    def Relabel(self,label_list):
        en_relabel_dic = {}
        de_relabel_dic = {}
        if len(self._label_list) >2:
            raise Exception("no more than two labels")
        label_length = int(np.ceil(np.log2(len(self._label_list))))
        for relabel, label in enumerate(self._label_list):
            en_relabel_dic[label] =int( bin(relabel)[2:].rjust(label_length, '0'))
            de_relabel_dic[int(bin(relabel)[2:].rjust(label_length, '0'))] = label
        return en_relabel_dic,de_relabel_dic

    # real labels to 0, 1
    def en_label(self,labels:Union[list,np.ndarray] =None, label_dic:dict=None):
        relabel = []
        if label_dic is None:
            raise Exception("input label_dic")
        for label in labels:
            relabel.append(label_dic[label])
        return np.asarray(relabel)
    # 0, 1 to real labels

    def de_label(self,labels:Union[list,np.ndarray] =None, label_dic:dict=None):
        origin_label = []
        if label_dic is None:
            label_dic =self._de_relabel_dic
        for label in labels:
            origin_label.append(label_dic[label])
        return origin_label


    def Sample_and_reduce(self,pics: np.ndarray = None):
        features = []
        if pics is None:
            raise Exception('input image for sampling')
        else:
            size = pics.shape
            if len(size) == 3:
                _number_of_pic = size[0]
                _height = size[1]
                _width = size[2]
            elif len(size) == 2:
                _number_of_pic = -1
                _height = size[0]
                _width = size[1]
            else:
                raise Exception('请传入一张或多张二维图像')
            center_x = int(np.ceil(_width / 2))
            center_y = int(np.ceil(_height / 2))
            if self._distance is None:
                #采样点位置：中间、上左、上中、上右、中左、下右、下中、下左、中右
                points = [(center_x, center_y),
                          (int(np.ceil(center_x / 2)), int(np.ceil(center_y / 2))),(center_x, int(np.ceil(center_y / 2))),
                          (int(np.ceil(_width - center_x / 2)), int(np.ceil(center_y / 2))),
                          (int(np.ceil(center_x / 2)), center_y),
                          (int(np.ceil(_width - center_x / 2)), int(np.ceil(_height - center_y / 2))),(center_x, int(np.ceil(_height - center_y / 2))),
                          (int(np.ceil(center_x / 2)), int(np.ceil(_height - center_y / 2))), (int(np.ceil(_width - center_x / 2)), center_y)]

                # print(points)

            else:
                if isinstance(self._distance,int):
                    x_distance = self._distance
                    y_distance = self._distance
                elif  isinstance(self._distance,list):
                    x_distance = self._distance[0]
                    y_distance = self._distance[1]
                points = [(center_x, center_y),
                          (int(np.ceil(center_x / 2)), int(np.ceil(center_y / 2))),
                          (center_x, int(np.ceil(center_y / 2))),
                          (int(np.ceil(_width - center_x / 2)), int(np.ceil(center_y / 2))),
                          (int(np.ceil(center_x / 2)), center_y),
                          (int(np.ceil(_width - center_x / 2)), int(np.ceil(_height - center_y / 2))),
                          (center_x, int(np.ceil(_height - center_y / 2))),
                          (int(np.ceil(center_x / 2)), int(np.ceil(_height - center_y / 2))),
                          (center_x, int(np.ceil(_height - center_y / 2)))]

        if _number_of_pic > 0:
            for pic_index in range(_number_of_pic):
                pic = pics[pic_index, :, :]
                pic_features = []
                for point in points:
                    sampling_block = pic[point[0] - self._radius:point[0] + self._radius+1, point[1] - self._radius:point[1] + self._radius+1]
                    pic_features.append(sampling_block.sum() / ((2*self._radius+1)** 2))
                features.append(pic_features)
            return np.asarray(features)
        else:
            # print("one image")
            pic_features = []
            for index, point in enumerate(points):
                sampling_block = pics[point[0] - self._radius:point[0] + self._radius+1, point[1] - self._radius:point[1] + self._radius+1]
                # average pooling
                pic_features.append(sampling_block.sum() / ((2*self._radius) ** 2))
            features.append(pic_features)
            return np.asarray(features)

    # solving intersections of two Gaussian functions
    def solve(self,m1, m2, std1, std2):
        a = 1 / (2 * std1 ** 2) - 1 / (2 * std2 ** 2)
        b = m2 / (std2 ** 2) - m1 / (std1 ** 2)
        c = m1 ** 2 / (2 * std1 ** 2) - m2 ** 2 / (2 * std2 ** 2) - np.log(std2 / std1)
        return np.roots(np.nan_to_num([a, b, c]))

    #gray value to binary 0 or 1
    def Trans_gray_to_label(self,feature: np.ndarray = None, intersections: list = None, targets: list = None):
        new_features = np.zeros(shape=feature.shape)
        for index, intersection in enumerate(intersections):
            target = targets[index]
            if isinstance(intersection, tuple):
                for i in range(new_features.shape[0]):
                    if (np.abs(feature[i, index] - intersection[0]) < np.abs(feature[i, index] - intersection[1])):
                        new_features[i, index] = 0
                    else:
                        new_features[i, index] = 1
            else:
                for i in range(new_features.shape[0]):
                    if feature[i, index] < intersection:
                        if target:
                            new_features[i, index] = 0
                        else:
                            new_features[i, index] = 1
                    else:
                        if target:
                            new_features[i, index] = 1
                        else:
                            new_features[i, index] = 0
        return new_features


    def _Intersection(self,mean_0: list, mean_1: list, std_0: list, std_1: list):
        intersections = []
        for index, value in enumerate(zip(mean_0, mean_1, std_0, std_1)):
            count = 0
            intersection = self.solve(value[0], value[1], value[2], value[3])
            for res in intersection:
                if (value[0] <= res <= value[1]) | (value[1] <= res <= value[0]):
                    intersections.append(res)
                    count += 1
            if count == 0:
                intersections.append((value[0], value[1]))
            elif count > 1:
                print('can not exist more than two intersections')
            else:
                pass
        return intersections

    # interpret the output of the circuit
    def interpret(self,features: Union[list, np.ndarray] = None, target: int = None):
        val = 0
        bit_strings = list(features)
        bit_strings.append(target)
        index = 0
        for bit_value in bit_strings[::-1]:
            val += bit_value * np.power(2, index)
            index += 1
        return int(val)

    def Bayesian_circuit(self,structure: list = None) -> Circuit:
        node_deep = np.asarray(deep_of_node(structure)).astype(int)
        structure = tans_to_parent_tree(structure)
        number_qubits = len(structure)

        if structure == None:
            raise Exception("Input structures")
        #X0
        cir = Circuit()
        count = 0
        cir += RY(f"theta{count}").on(0)
        count += 1
        max_deep = node_deep.max()

        is_needed = node_deep == 0
        items = len(is_needed)

        for item in range(items):
            if is_needed[item] == True:
                cir += RY(f"theta{count}").on(number_qubits-item,0)
                count += 1
        cir += X.on(0)
        for item in range(items):
            if is_needed[item] == True:
                cir += RY(f"theta{count}").on(number_qubits-item,0)
                count += 1
        cir += X.on(0)
        for current_deep in range(1, max_deep + 1):
            father_list = []
            is_needed = node_deep == current_deep
            items = len(is_needed)
            for item in range(items):
                if is_needed[item] == True:
                    father = number_qubits - structure[item][0]
                    if father in father_list:
                        continue
                    else:
                        father_list.append(father)
            for item in range(items):
                if is_needed[item] == True:
                    cir += RY(f"theta{count}").on(number_qubits -item, [number_qubits -structure[item][0], 0])
                    count += 1

            # 01
            cir += X.on(0)
            for item in range(items):
                if is_needed[item] == True:
                    cir += RY(f"theta{count}").on(number_qubits -item, [number_qubits -structure[item][0], 0])
                    count += 1
            # 00
            for index in father_list:
                cir += X.on(index)
            for item in range(items):
                if is_needed[item] == True:
                    cir += RY(f"theta{count}").on(number_qubits - item, [number_qubits -structure[item][0], 0])
                    count += 1
            #10
            cir += X.on(0)
            for item in range(items):
                if is_needed[item] == True:
                    cir += RY(f"theta{count}").on(number_qubits -item, [number_qubits -structure[item][0], 0])
                    count += 1
            # 11
            for index in father_list:
                cir += X.on(index)
        return cir

    def Probability_statistic(self,features: np.ndarray = None, label: np.ndarray = None, structure: list = None):
        if features is None or label is None:
            raise Exception("Input data")
        if structure == None:
            raise Exception("Input data structure")

        probability = {}
        count = 0

        count_1 = len(label[label == 1])
        count_0 = len(label[label == 0])
        probability[f"theta{count}"] = (count_0 +1 )/ (count_0 + count_1 +len(self._label_list))
        count += 1

        node_deep = np.asarray(deep_of_node(structure)).astype(int)
        structure = tans_to_parent_tree(structure)
        is_needed = node_deep == 0
        items = len(is_needed)
        for item in range(items):
            if is_needed[item] == True:
                probability[f"theta{count}"] = (np.sum(features[label==1,item]==0)+1)/(count_1+2)
                count +=1
        for item in range(items):
            if is_needed[item] == True:
                probability[f"theta{count}"] = (np.sum(features[label==0,item]==0)+1)/(count_0+2)
                count +=1
        max_deep = node_deep.max()
        for current_deep in range(1, max_deep + 1):
            father_list = []
            is_needed = node_deep == current_deep
            items = len(is_needed)
            for item in range(items):
                if is_needed[item] == True:
                    father = structure[item][0]
                    if father in father_list:
                        continue
                    else:
                        father_list.append(father)
            #11
            for item in range(items):
                if is_needed[item] == True:
                    probability[f"theta{count}"] = (
                        np.sum((features[label == 1, item] == 0) & (features[label == 1,structure[item][0]] == 1) )+ 1) / (np.sum(features[label == 1,structure[item][0]] == 1) + 2)
                    count += 1
            for item in range(items):
                if is_needed[item] == True:
                    probability[f"theta{count}"] = (
                        np.sum((features[label == 0, item] == 0) & (features[label == 0, structure[item][0]] == 1)) + 1) / (np.sum(features[label == 0,structure[item][0]] == 1) + 2)
                    count += 1
            for item in range(items):
                if is_needed[item] == True:
                    probability[f"theta{count}"] = (np.sum((features[label ==0 , item] == 0) & (features[label == 0, structure[item][0]] ==0)) + 1) / (np.sum(features[label == 0, structure[item][0]] == 0) + 2)
                    count += 1
            for item in range(items):
                if is_needed[item] == True:
                    probability[f"theta{count}"] = (np.sum((features[label == 1, item] == 0) & (features[label == 1,structure[item][0]] == 0)) + 1) / (np.sum(features[label == 1, structure[item][0]] == 0) + 2)
                    count += 1
        return probability

    def Trans_pr_to_para(self,probability):
        parameters = []
        for theta in probability:
            probability[theta] = 2 * np.arccos(np.sqrt(probability[theta]))
            parameters.append(probability[theta])
        # print(parameters)
        return probability

    def _check_fit(self):
        if self._probability_table is None:
            raise Exception('Training first')
        else:
            return True

    def predict(self, X: np.ndarray = None):
        if X is None:
            raise Exception('Input data for prediction')
        if self._check_fit():
            features_test = self.Sample_and_reduce(X)
            features = self.Trans_gray_to_label(feature=features_test, intersections=self._intersections,
                                                targets=self._targets)
            result = []

            for number in range(features.shape[0]):
                index = self.interpret(features[number,], 0)
                p_0 = self._probability_table[index]
                index = self.interpret(features[number,], 1)
                p_1 = self._probability_table[index]
                if p_0 > p_1:
                    result.append(0)
                else:
                    result.append(1)
            result = self.de_label(result)
            return result


    def fit(self):
        set = self.set
        label_list = self._label_list
        if len(label_list) != 2:
            raise Exception("Input two classes")

        structure = self._structure
        circuit = self._circuit

        en_dic, de_dic = self.Relabel(label_list)

        self._en_relabel_dic,self._de_relabel_dic = en_dic,de_dic

        x_train, y_train, x_test, y_test = self.Get_dataset(set, label_list=label_list)

        features = self.Sample_and_reduce(x_train)

        label = self.en_label(y_train, label_dic=en_dic)

        features_average_0 = np.average(features[label == 0], axis=0)
        features_standard_0 = np.std(features[label == 0], axis=0)
        features_average_1 = np.average(features[label == 1], axis=0)
        features_standard_1 = np.std(features[label == 1], axis=0)

        _intersections = self._Intersection(features_average_0, features_average_1, features_standard_0, features_standard_1)
        self._intersections = _intersections

        _targets = features_average_0 < features_average_1
        self._targets = _targets

        new_features = self.Trans_gray_to_label(feature=features, intersections=_intersections, targets=_targets)

        probability= self.Probability_statistic(new_features, label, structure)

        cir_parameters = self.Trans_pr_to_para(probability)

        sim = Simulator('mqvector', circuit.n_qubits)
        sim.apply_circuit(circuit, cir_parameters)

        probability_table = np.real(sim.get_qs() ** 2)
        self._probability_table = probability_table

        result = self.predict(x_test)

        acc = np.sum(result == y_test) / len(y_test) * 100 / 100
        result = np.asarray(result)
        y_test = np.asarray(y_test)
        tp = np.sum((result == self._label_list[0]) & (y_test == self._label_list[0]))
        fn = np.sum((result == self._label_list[1]) & (y_test == self._label_list[0]))
        fp = np.sum((result == self._label_list[0]) & (y_test == self._label_list[1]))
        tn = np.sum((result == self._label_list[1]) & (y_test == self._label_list[1]))

        return acc,(tp,fn,fp,tn)


def get_one_pair_acc(set,label_list,radius,structure=None):
    classifier = QBCs(set = set, label_list=label_list,structure = structure,radius=radius)
    acc,property = classifier.fit()
    print(acc)
    # print(np.asarray(property))
    return acc

def get_structure_by_name(kind,set ="mnist"):
    if kind == "nb":
        structure = [[], [], [], [], [], [], [], [], []] #no child node
    elif kind == "spode":
        structure = [[1, 2, 3, 4, 5, 6, 7, 8], [], [], [], [], [], [], [], []] #x1 has child node 2,3,,9
    elif kind == "tan":
        if set == "mnist":
            structure = [[1, 2, 4, 6, 7, 8], [5], [], [], [], [], [], [3], []]
        elif set == "fashion_mnist":
            structure = [[2, 4, 6, 8], [], [], [], [], [], [7], [1], [3, 5]]
    elif kind == "symmetric":
        structure = [[],[5],[6],[7],[8],[],[],[],[]] #x2->x6, x3->x7, x4->x8, x5->x9
    else:
        structure = [[], [], [], [], [], [], [], [], []]#naive
    return structure

def main():

    #set = "mnist"
    set = "fashion_mnist"

    #method = 'nb'
    #method = 'spode'
    method = 'tan'
    #method = 'symmetric'

    # block_radius = (0, 1, 2, 3, 4, 5)  #1*1,3*3,5*5,7*7,9*9,11*11
    block_radius = (3,)

    for radius in block_radius:
        acc_matrix = np.zeros(shape=(10, 10))
        result_list = np.zeros(shape=(45))
        i = 0
        for first_label in range(0, 9):
            for second_label in range(first_label + 1, 10):
                structure = get_structure_by_name(method, set)
                acc = get_one_pair_acc(set,label_list=[first_label, second_label],radius=radius,structure=structure)
                acc_matrix[first_label, second_label] = acc
                result_list[i] = acc_matrix[first_label, second_label]
                i = i + 1

        print("type:", method, "dataset:", set, "Radius=", radius)
        print(result_list)
        print("avg=", np.average(result_list))

if __name__ == '__main__':
    main()

