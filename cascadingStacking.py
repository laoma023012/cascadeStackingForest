# -*- coding: utf-8 -*-
import numpy as np
from stacking import stacking
from standard_template import *
from sklearn.cross_validation import KFold

class cacsadeStacking(stacking,standard_classification_template):
    
    def __init__(self, max_iterations =  1 ):
        self.max_iterations = max_iterations
        super(cacsadeStacking, self).__init__()
        self.iterations = str(14)
        self.Kfold = 3
        pass
    
    
    def cascadeForestTestProcess(self,data):
        for index in range(self.max_iterations):
            print(' stage -'+ str(index) + 'test')
            self.iterations = str(index)
            data = stacking.stackTestProcess(self,data)
        return data
        pass
        
    def cascadeForestTrainProcess(self,data,label):
        #model = standard_classification_template()
        '''
        差 model 名字随迭代次数改进
        '''
        for index in range(self.max_iterations):
            print(' stage -'+ str(index) + 'train')
            self.iterations = str(index)
            stacking.stackStage1Train(self,data,label)
            data, label = stacking.stackStage1FeatureAndLabel(self,data,label)
            data = np.array(data)
            label = np.array(label)
    
    def cascadeForestTrain(self,dataPath,labelPath):
        data,label = dataPreprocessing.dataTrainPort(self,dataPath,labelPath)
        self.cascadeForestTrainProcess(data,label)
        
        #model = standard_classification_template()
        '''
        差 model 名字随迭代次数改进
        '''
        #for index in range(self.max_iterations):
        #    print(' stage -'+ str(index) + 'train')
        #    self.iterations = str(index)
        #    stacking.stackStage1Train(self,data,label)
        #    data, label = stacking.stackStage1FeatureAndLabel(self,data,label)
        #    data = np.array(data)
        #    label = np.array(label)
            
        '''        
        print(' stage -1 ')
        #super(cacsadeStacking,model).iterDefine(2)
        stacking.stackStage1Train(self,data,label)
        data, label = stacking.stackStage1FeatureAndLabel(self,data,label)
        
        data = np.array(data)
        label = np.array(label)
        
        print(' stage -2 ')
       
        stacking.stackStage1Train(self,data,label)
        data, label = stacking.stackStage1FeatureAndLabel(self,data,label)
        
        data = np.array(data)
        label = np.array(label)
        
        print(' stage -3 ')
       
        stacking.stackStage1Train(self,data,label)
        data, label = stacking.stackStage1FeatureAndLabel(self,data,label)
        pass
        '''
    
    def cascadeForestTest(self,dataPath):
        data = dataPreprocessing.dataTestPort(self,dataPath)
        return self.cascadeForestTestProcess(data)
        
        '''
        for index in range(self.max_iterations):
            print(' stage -'+ str(index) + 'test')
            self.iterations = str(index)
            data = stacking.stackTestProcess(self,data)
        return data
        pass
        '''
    
    def cascadeForestEval(self,dataPath,labelPath):
        data, label = dataPreprocessing.dataEvalPort(self, dataPath, labelPath)
        data = self.dataNormalized(data)        
        kf = KFold(len(label),n_folds = self.Kfold, random_state = self.random_state)
        
        foldCount = 1
        score = []
        for trainIndex, testIndex in kf:
            print('this is the',foldCount,'th fold for cascadeForestEval')
            dataTrain, dataTest, labelTrain, labelTest = data[trainIndex], data[testIndex], label[trainIndex], label[testIndex]
            
            self.cascadeForestTrainProcess(dataTrain,labelTrain)
            probModel = self.cascadeForestTestProcess(dataTest)
            
            TopPreds = standard_classification_template.TopPredict(self,probModel)
            tmpScore = standard_classification_template.showEvalResult(self, TopPreds, labelTest)
            score.append(tmpScore)
            foldCount = foldCount + 1
        pass
        avgScore = sum(score)/len(score)
        print('k - fold score is ',avgScore)
        pass
    
if __name__ == "__main__":
    dataPath = 'D:/2017_8_8_task/20170808/SAR/task_classification/comp_dataset/fc7.txt'
    labelPath = 'D:/2017_8_8_task/test/Stacking_test/labelinfo.txt'
    model = cacsadeStacking()
    #model.cascadeForestTrain(dataPath,labelPath)
    #result = model.cascadeForestTest(dataPath)
    model.cascadeForestEval(dataPath,labelPath)
    pass
    