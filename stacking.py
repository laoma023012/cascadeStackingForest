# -*- coding: utf-8 -*-
import numpy as np
from standard_template import *
from external_template import gbdtMethod
from sklearn import linear_model
from sklearn.externals import joblib
from sklearn.cross_validation import train_test_split

selectMethod = {'svm': 1 , 'lr': 1 , 'nb': 1 , 'rf': 0 , 'gbdt': 0 }

class stacking(svmMethod,lrMethod,nbMethod,rfMethod):
    def __init__(self):
        self.test_size = 0.3
        self.random_state = 48
    
    def establishStage2Model(self):
        logreg = linear_model.LogisticRegression(penalty='l2', dual = False, C = 50, solver='newton-cg',max_iter=1000)
        return logreg
    
    def stackTrainTestSplit(self,dataPath,labelPath):
        data,label = dataPreprocessing.dataStackPort(self,dataPath,labelPath)
        dataTrain, dataTest, labelTrain, labelTest = train_test_split(data, label, test_size = self.test_size, random_state = self.random_state)
        return dataTrain, dataTest, labelTrain, labelTest
        pass
    
    def stackStage2TestProcess(self,data):
        model = joblib.load( 'stage2model' )
        probModel = model.predict_proba(data)
        TopPreds = super().TopPredict(probModel)
        return TopPreds,probModel
        pass
    
    def stackStage2TrainProcess(self,data,label):
        print('start stage2 training')
        model = self.establishStage2Model()
        model.fit(data,label)
        joblib.dump(model, 'stage2model' )
        pass
    
    def stackTestProcess(self,data):
        '''
        调用每个基类的初始化函数
        '''
        probTestSet = []
        
        svmMethod.__init__(self)
        svmResult = svmMethod.stackingTestPort(self,data)
        probTestSet.append(svmResult)
        
        lrMethod.__init__(self)
        lrResult = lrMethod.stackingTestPort(self,data)
        probTestSet.append(lrResult)
        
        nbMethod.__init__(self)
        nbResult = nbMethod.stackingTestPort(self,data)
        probTestSet.append(nbResult)
        
        rfMethod.__init__(self)
        rfResult = rfMethod.stackingTestPort(self,data)
        probTestSet.append(rfResult)
        
        prob = self.stackingPlan(probTestSet)
        return prob
        pass
    
    
    def stackTrainProcess(self,data,label):
        '''
        调用每个基类的初始化函数
        '''
        svmMethod.__init__(self)
        svmMethod.stackingTrainPort(self,data,label)
        lrMethod.__init__(self)
        lrMethod.stackingTrainPort(self,data,label)
        nbMethod.__init__(self)
        nbMethod.stackingTrainPort(self,data,label)
        rfMethod.__init__(self)
        rfMethod.stackingTrainPort(self,data,label)
        pass
    
    def stackingPlan(self,probStackSet):
        '''
        example:
                -------------------------------------------------------------
                base learner -- svm  --  lr  --   nb    --  rf  --   gbdt --      
                probability   prob 1 ,, prob 2 ,, prob 3 ,, prob 4 ,, prob 5
                -------------------------------------------------------------
            
        '''
        '''
        average plan
              
               prob =  (prob 1 + ... + prob N) / N
        '''
        ''' PLAN 1
        index = 1
        prob = 0
        for probSet in probStackSet:
            probSet = np.array(probSet)
            prob = prob + probSet
            index = index + 1
        return prob / index
        '''
        return probStackSet[1]
        pass
    
    def stackStage1FeatureAndLabelProcess(self,data,label):
        '''
        调用每个基类的初始化函数
        '''
        probStackSet = []
        svmMethod.__init__(self)
        '''
        为了模块间的松耦合性 将 stack 的单模型结果 写成 list append 型
        '''
        
        '''
        svm result 
        '''
        probSet,labelSet = svmMethod.stackingStage1FeatureAndLabel(self,data,label)
        probStackSet.append(probSet)
        
        '''
        
        lr result 
        '''
        lrMethod.__init__(self)
        probSet,labelSet = lrMethod.stackingStage1FeatureAndLabel(self,data,label)
        probStackSet.append(probSet)
        
        '''
        nb result 
        '''
        nbMethod.__init__(self)
        probSet,labelSet = nbMethod.stackingStage1FeatureAndLabel(self,data,label)
        probStackSet.append(probSet)
        
        '''
        rf result 
        '''
        rfMethod.__init__(self)
        probSet,labelSet = rfMethod.stackingStage1FeatureAndLabel(self,data,label)
        probStackSet.append(probSet)
        
        return probStackSet, labelSet
        pass
    
    def stackStage1FeatureAndLabel(self,data,label):
        #data,label = dataPreprocessing.dataStackPort(self,data,label)
        probStackSet,labelSet = self.stackStage1FeatureAndLabelProcess(data,label)
        '''
        execute stack plan
        '''
        probStackPlanResult = self.stackingPlan(probStackSet)
        return probStackPlanResult,labelSet
        
    def stackStage1Train(self,dataPath,labelPath):
        try :
            data,label = dataPreprocessing.dataStackPort(self,dataPath,labelPath)
        except ValueError:
            data = dataPath
            label = labelPath
        self.stackTrainProcess(data, label)
        pass
    
    def stackStage1Test(self,dataPath):
        data = dataPreprocessing.dataTestPort(self,dataPath)
        return self.stackTestProcess(data)
       
    def stackStage2Train(self,dataPath,labelPath):
        probStackPlanResult,labelSet = self.stackStage1FeatureAndLabel(dataPath,labelPath)
        '''
        probStackPlanResult : stage 2 train dataset
        labelSet : stage 2 train label
        '''
        self.stackStage2TrainProcess(probStackPlanResult,labelSet)
        pass
    
    def stackStage2Test(self,dataPath):
        prob = self.stackStage1Test(dataPath)
        TopPredict = self.stackStage2TestProcess(prob)
        return TopPredict
        pass
     
    #def stackEval(self,dataPath)    
        
if __name__ == "__main__":
    dataPath = 'D:/2017_8_8_task/20170808/SAR/task_classification/comp_dataset/fc7.txt'
    labelPath = 'D:/2017_8_8_task/test/Stacking_test/labelinfo.txt'
    ''' 单元测试 '''
    model = stacking()
    ''' 第一阶段训练 单元测试 '''
    model.stackStage1Train(dataPath,labelPath)
    ''' 第二阶段训练 单元测试 '''
    model.stackStage2Train(dataPath,labelPath)
    ''' 第一阶段 和 第二阶段 单元测试'''
    TopPredict = model.stackStage2Test(dataPath)
    ''' 获取 第一阶段 得到的 feature 和 label '''
    probStackPlanResult,labelSet = model.stackStage1FeatureAndLabel(dataPath,labelPath)