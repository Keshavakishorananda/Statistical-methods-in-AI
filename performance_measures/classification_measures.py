import numpy as np

class Measures:
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred

    def find_confusion_matrix(self):
        self.classes = np.unique(self.y_true)
        
        confusion_matrix = []
        for i in self.classes:
            row = []
            TP = 0
            TN = 0
            FP = 0
            FN = 0
            for x,y in zip(self.y_true, self.y_pred):
                if x == i and y == i:
                    TP += 1
                elif x == i and y != i:
                    FN += 1
                elif x != i and y == i:
                    FP += 1
                else:
                    TN += 1
            row.append(TP)
            row.append(FN)
            row.append(FP)
            row.append(TN)

            confusion_matrix.append(row)

        return confusion_matrix
        
    def accuracy(self):
        return np.mean(self.y_true == self.y_pred)

    def precision_micro(self):
        matrix = self.find_confusion_matrix()
        total_TP = 0
        total_FP = 0
        for i in matrix:
            total_TP += i[0]
            total_FP += i[2]
        return total_TP / (total_TP + total_FP)

    def precision_macro(self):
        matrix = self.find_confusion_matrix()
        # LLM prompt : How to caluclate precision when denominator is zero?
        # start code
        precision = 0
        reduced_num = 0
        for i in matrix:
            if i[0] + i[2] == 0:
                reduced_num += 1
                precision += 0
            else:
                precision += (i[0] / (i[0] + i[2]))
        return precision / (len(matrix)-reduced_num)
        # end code
        

    def recall_micro(self):
        matrix = self.find_confusion_matrix()
        total_TP = 0
        total_FN = 0
        for i in matrix:
            total_TP += i[0]
            total_FN += i[1]
        return total_TP / (total_TP + total_FN)

    def recall_macro(self):
        matrix = self.find_confusion_matrix()
        recall = 0
        reduced_num = 0
        for i in matrix:
            if i[0] + i[1] == 0:
                reduced_num += 1
                recall += 0
            else:
                recall += (i[0] / (i[0] + i[1]))
        return recall / (len(matrix)-reduced_num)
        

    def f1_score_micro(self):
        return 2 * (self.precision_micro() * self.recall_micro()) / (self.precision_micro() + self.recall_micro())


    def f1_score_macro(self):
        return 2 * (self.precision_macro() * self.recall_macro()) / (self.precision_macro() + self.recall_macro())