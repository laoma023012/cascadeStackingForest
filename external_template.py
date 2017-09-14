# -*- coding: utf-8 -*-
import numpy as np
import xgboost as xgb
from standard_template import dataPreprocessing
from sklearn.cross_validation import KFold

class gbdtMethod(dataPreprocessing):
    
    def __init__(self):
        pass
    
    def TopSelect(self, SortRank, N):
       TopSelect = SortRank[0:N]
       return TopSelect
   
    def TopPredict(self,preds):
       TopPreds = []
       for index in preds:
           SortRank = np.argsort(-index)
           TopN = self.TopSelect(SortRank, 2)
           TopPreds.append(TopN)
       return TopPreds
    
    def jugResult(self, preds, trueLabel):
        for num in preds:
            if num == trueLabel:
                return True
        return False 
    
    def showEvalResult(self, Top2Preds, labelTest):
        totalNum = len(labelTest)
        rightNum = 0
        for preds, trueLabel in zip(Top2Preds, labelTest):
            jugTag = self.jugResult(preds, trueLabel)
            if jugTag == True:
                rightNum = rightNum + 1
        print('Precision = ', rightNum/totalNum)        
        return rightNum/totalNum
        pass
    
    def paramSetting(self,label):
        param = {
                 'max_depth':20,
                 'min_child_weight':12,
                 'learning_rate':0.1,
                 'eta':0.01,
                 'subsample':1,
                 'silent':0,
                 #'reg_alpha':'l2',
                 'num_class':len(set(label)),
                 'objective':'multi:softprob',
                 #'eval_matric':'map'
               }
        return param
    
    def establishModel(self,data,label):
        dtrain = xgb.DMatrix(data = np.array(data), label = np.array(label))
        return dtrain
    
    '''
    ease for re-use in eval 
    '''
    
    def trainProcess(self,data,label):
        dtrain = self.establishModel(data = data, label = label)
        param = self.paramSetting(label)
        num_round = 51
        bst = xgb.train(param, dtrain, num_round)
        bst.save_model('model')
        pass
    
    def testProcess(self,data):
        model = xgb.Booster(model_file = 'model')
        preds = model.predict(xgb.DMatrix(np.array(data)))
        TopPreds = self.TopPredict(preds)
        return TopPreds
        pass
    
    def fiveFoldEval(self,dataPath,labelPath):
        data,label = dataPreprocessing.dataEvalPort(self, dataPath, labelPath)
        kf = KFold(len(label),n_folds = 5)
        '''
        func : describe the 5-fold cross-valdation
        mean: for each fold we describe it
        '''
        foldCount = 1
        score = []
        for trainIndex, testIndex in kf:
            print('this is the ',foldCount,'th fold')
            dataTrain, dataTest, labelTrain, labelTest = data[trainIndex], data[testIndex], label[trainIndex], label[testIndex]
            self.trainProcess(dataTrain,labelTrain)
            TopPreds = self.testProcess(dataTest)
            tmpScore = self.showEvalResult(TopPreds, labelTest)
            score.append(tmpScore)
            foldCount = foldCount + 1
        pass
        avgScore = sum(score)/len(score)
        print('k - fold score is ',avgScore)
        pass
    
    def test(self,dataPath):
        data = dataPreprocessing.dataTestPort(self, dataPath)
        print(type(data))
        TopPreds = self.testProcess(data)
        return TopPreds
    
    def train(self,dataPath,labelPath):
        data,label = dataPreprocessing.dataTrainPort(self, dataPath, labelPath)
        self.trainProcess(data,label)
        pass

if __name__ == '__main__':
    dataPath = 'D:/2017_8_8_task/20170808/SAR/task_classification/comp_dataset/fc7.txt'
    labelPath = 'D:/2017_8_8_task/test/Stacking_test/labelinfo.txt'
    model = gbdtMethod()
    model.train(dataPath,labelPath)
    model.fiveFoldEval(dataPath,labelPath)
