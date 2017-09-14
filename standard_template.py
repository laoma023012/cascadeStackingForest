# -*- coding: utf-8 -*-
import numpy as np
import warnings
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.cross_validation import KFold


ITERATION = 5

class dataPreprocessing(object):
    def __init__(self):
        #self.iterations = str(iterations)
        pass
    
    '''
    func : loadEvalOrTrainData
    
    Input : dataPath, labelPath with different format
    
    Meaning: To support the different format of data like txt json or xml 
    
    Output : data / label with the format of numpy
    
    '''
    '''
    把迭代次数的定义 放在这是为了以后 额外的模板 和 标准模板都有最原始的基类
    '''
    
    def loadXmlData(self):
        '''
        pre-load the xml foramt transform to numpy
        '''
        pass
    
    def loadJsonData(self):
        '''
        pre-load the json foramt transform to numpy
        '''
        pass
    
    def loadTxtData(self,dataPath,labelPath):
        data = np.loadtxt(dataPath)
        label = np.loadtxt(labelPath,dtype='int32')
        return data, label
        pass
        
    def dataTestPort(self,dataPath):
        # --- 忽略 labelPath 的警告 ------
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data,label = self.loadTxtData(dataPath,[])
        return data
        pass
    
    def dataEvalPort(self,dataPath,labelPath):
        data, label = self.loadTxtData(dataPath,labelPath)
        return data, label
        pass
        
    def dataTrainPort(self,dataPath,labelPath):
        data, label = self.loadTxtData(dataPath,labelPath)
        return data, label
        pass
    
    def dataStackPort(self,dataPath,labelPath):
        data, label = self.loadTxtData(dataPath,labelPath)
        return data, label
        pass


'''
func: standard_classification_template

Meaning: Ease for use of sklearn-style model
         
Introduction:
        train : train this model with data and label
         
        test : test this model with data
        
        eval : K - fold eval this model with data and label 

'''
class standard_classification_template(dataPreprocessing):
    def __init__(self):
        #super(standard_classification_template,self).__init__()
        #self.iterations = str(1)
        self.stageName = 'stage -'
        self.modelName = 'model'
        self.methodName = 'svm'
        self.random_state = 48
        self.Kfold = 5
        pass
    
    
    def iterDefine(self):
        self.iterations = str(100)
    
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
    
    def jugResult(self,preds, trueLabel):
        for num in preds:
            if num == trueLabel:
                return True
        return False   
    
    def TopSelect(self, sortRank, N):
       TopSelect = sortRank[0:N]
       return TopSelect
   
    def TopPredict(self,preds):
       TopPreds = []
       for index in preds:
           SortRank = np.argsort(-index)
           TopN = self.TopSelect(SortRank, 2)
           TopPreds.append(TopN)
       return TopPreds
    
    def dataNormalized(self,data):
        data = np.transpose(data)
        X_scaled = preprocessing.scale(data)
        X_scaled = np.transpose(X_scaled)
        return X_scaled
        pass
    
    def establishModel(self):
        model = SVC(C = 0.01, kernel='linear',shrinking=True,decision_function_shape='ovo',random_state= 48,probability=True)
        return model
        pass
    
    def testProb(self,data):
        data = self.dataNormalized(data)
        try:
            model = joblib.load( self.modelName )
        except FileNotFoundError:
            print('Hey man, model not found , please train this model first！')
            return
            
        probModel = model.predict_proba(data)
        return probModel
        pass
    
    def testProcess(self,data):
        probModel = self.testProb(data)
        TopPreds = self.TopPredict(probModel)
        return TopPreds
        pass
    
    def trainProcess(self,data,label):
        data = self.dataNormalized(data)
        model = self.establishModel()
        model.fit(data,label)
        print(self.modelName)
        joblib.dump(model, self.modelName )
               
    
    def train(self,dataPath,labelPath):
        '''
        基于树的模型不用做归一化，基于统计的模型需要归一化
        '''
        data,label = dataPreprocessing.dataTrainPort(self, dataPath, labelPath)
        self.trainProcess(data,label)
        pass
    
    def test(self,dataPath):      
        data = dataPreprocessing.dataTestPort(self, dataPath)
        return self.testProcess(data)
    
    def KFoldEval(self,dataPath,labelPath):
        data, label = dataPreprocessing.dataEvalPort(self, dataPath, labelPath)
        data = self.dataNormalized(data)
        
        kf = KFold(len(label),n_folds = self.Kfold, random_state = self.random_state)
        '''
        func : describe the 5-fold cross-valdation
        mean: for each fold we describe it
        '''
        foldCount = 1
        score = []
        for trainIndex, testIndex in kf:
            print('this is the',foldCount,'th fold')
            dataTrain, dataTest, labelTrain, labelTest = data[trainIndex], data[testIndex], label[trainIndex], label[testIndex]
            model = self.establishModel()
            model.fit(dataTrain,labelTrain)
            probModel = model.predict_proba(dataTest)
            TopPreds = self.TopPredict(probModel)
            tmpScore = self.showEvalResult(TopPreds, labelTest)
            score.append(tmpScore)
            foldCount = foldCount + 1
        pass
        avgScore = sum(score)/len(score)
        print('k - fold score is ',avgScore)
        
    def stackingStage1FeatureAndLabel(self,data,label):
        
        #print('data',len(data))
        #print('label',len(label))
        
        #self.iterDefine()
        
        #data, label = dataPreprocessing.dataEvalPort(self, data, label)
        '''
        Test with K trained models and ensemble with the test data
        example:
                  k - 1 fold train     1 test                               test result
                                    '                                           '
                                    '               concat test result          '
                                    '                                           '
                  k - 1 fold train     1 test                               test result
                  
                  
        '''
        kf = KFold(len(label), n_folds = self.Kfold, random_state = self.random_state)
        foldCount = 1
        self.standardModelName = self.modelName
        probSet = []
        labelSet = []
        for trainIndex, testIndex in kf:
            #print('trainIndex',trainIndex)
            #print('testIndex',testIndex)
            
            print('this is the',foldCount,'th fold stacking to extract data and label')
            dataTrain, dataTest, labelTrain, labelTest = data[trainIndex], data[testIndex], label[trainIndex], label[testIndex]
            '''
            change the name while k-fold training processing
            
            '''
            #self.tmpModelName = self.methodName + self.modelName + str(foldCount)
            self.tmpModelName = self.stageName + self.iterations + self.methodName + self.modelName + str(foldCount)
            self.modelName = self.tmpModelName
            '''
            train process
            '''
            probModel = self.testProb(dataTest)
            
            probSet.extend(probModel)
            labelSet.extend(labelTest)
            '''
            需要把 probSet 再反转过来 现在是 5 4 3 2 1 变成 1 2 3 4 5
            '''
            self.modelName = self.standardModelName
            foldCount = foldCount + 1
        return probSet,labelSet
    
    def stackingTrainPort(self,data,label):
        #self.iterDefine()
        #try :
        #    data,label = dataPreprocessing.dataStackPort(self,dataPath,labelPath)
        #except ValueError:
        #    data = dataPath , label = labelPath
            
        '''
        example:
                  k - 1 fold train     1 test              trained model  1               
                                    '                             '                   
                                    '                             '
                                    '                             ' 
                  k - 1 fold train     1 test              trained model  N                
        '''
        
        '''
        stackTrain 和 stackTest 的数据集合不应是同一集合
        '''
        
        '''
        实际调用的时候没有 数据的文件传输借口
        '''
        #data, label = dataPreprocessing.dataEvalPort(self, data, label)
        '''
        data and label is numpy format data , not txt file
        '''
        kf = KFold(len(label), n_folds = self.Kfold, random_state = self.random_state)
        '''
        stackPort 分数据集的时候 需要 分相同的 K 份
        '''
        foldCount = 1
        self.standardModelName = self.modelName
        for trainIndex, testIndex in kf:
            #print('trainIndex',trainIndex)
            #print('testIndex',testIndex)
            print('this is the',foldCount,'th fold stacking')
            dataTrain, dataTest, labelTrain, labelTest = data[trainIndex], data[testIndex], label[trainIndex], label[testIndex]
            '''
            change the name while k-fold training processing
            
            '''
            self.tmpModelName = self.stageName + self.iterations + self.methodName + self.modelName + str(foldCount)
            self.modelName = self.tmpModelName
            '''
            train process
            '''
            self.trainProcess(dataTrain,labelTrain)
            self.modelName = self.standardModelName
            foldCount = foldCount + 1

    def stackingTestPort(self,data):
        
        #self.iterDefine()
        '''
        example:
            
            train :
                  k - 1 fold train     1 test              trained model  1               
                                    '                             '                   
                                    '                             '
                                    '                             ' 
                  k - 1 fold train     1 test              trained model  N  
                  
            test:        
                                       trained model 1     result 1
                                             '                '
                 testData              trained model m     result m
                                             '                '
                                       trained model n     result n
        
        '''
        
        '''
        stackTrain 和 stackTest 的数据集合不应是同一集合 它们的并集是完整集合 交集为0
        '''
        
        '''
        实际调用的时候没有 数据的文件传输借口
        '''
        #data = dataPreprocessing.dataTestPort(self, data)
        foldCount = 1
        self.standardModelName = self.modelName
        sumProb = 0
        for i in range(self.Kfold):
            #self.tmpModelName = self.methodName + self.modelName + str(foldCount)
            self.tmpModelName = self.stageName + self.iterations + self.methodName + self.modelName + str(foldCount)
            self.modelName = self.tmpModelName   
            #print('modelName',self.modelName)
            probModel = self.testProb(data)
            self.modelName = self.standardModelName
            foldCount = foldCount + 1
            try:
                sumProb = sumProb + probModel
            except TypeError:
                 print('Lost model name is ' + self.tmpModelName + '！')
                 return
        avgProb = sumProb / self.Kfold
        return avgProb
        pass

'''
func : reconstruct the code with inherit

'''    
class svmMethod(standard_classification_template):
    def __init__(self):
        super(svmMethod, self).__init__()
        self.methodName = 'svm'
        #self.iterations = ''
        pass
    
    def establishModel(self):
        #model = SVC(C = 0.01, kernel='linear',shrinking=True,decision_function_shape='ovo',random_state= 48,probability=True)
        model = SVC(C = 0.01, kernel='rbf',shrinking = True,decision_function_shape='ovo',random_state= 48,probability=True)
        return model
        pass
    '''
    为了增加可读性 需要再在子类添加一次接口的声明
    '''
    def train(self,dataPath,labelPath):
        print('start svm train')
        super().train(dataPath, labelPath)
    
    def test(self,dataPath):
        print('start svm test')
        return super().test(dataPath)
    
    def KFoldEval(self,dataPath,labelPath):
        super().KFoldEval(dataPath,labelPath)
       
    '''
    stack 相关的三个接口输入是 numpy 格式的 数据 方便上级接口调用 而不是 file 格式的输入
    ''' 
        
    def stackingTrainPort(self,data,label):
        super().stackingTrainPort(data,label)
        
    def stackingTestPort(self,data):
        return super().stackingTestPort(data)
    
    def stackingStage1FeatureAndLabel(self,data,label):
        probSet,labelSet = super().stackingStage1FeatureAndLabel(data,label)
        return probSet,labelSet

class lrMethod(standard_classification_template):
    def __init__(self):
        super(lrMethod, self).__init__()
        self.methodName = 'lr'
        #self.iterations = ''
        pass
    
    def establishModel(self):
        model = linear_model.LogisticRegression(penalty='l2', dual = False, C = 50, solver='newton-cg',max_iter=1000)
        #model = linear_model.LogisticRegression( dual = False, C = 0.01, solver='newton-cg',max_iter=1000)
        return model
        pass
    
    '''
    为了增加可读性 需要再在子类添加一次接口
    '''
    def train(self,dataPath,labelPath):
        print('start lr train')
        super().train(dataPath, labelPath)
    
    def test(self,dataPath):
        print('start lr test')
        return super().test(dataPath)
    
    def KFoldEval(self,dataPath,labelPath):
        super().KFoldEval(dataPath,labelPath)
 
    '''
    stack 相关的三个接口输入是 numpy 格式的 数据 方便上级接口调用 而不是 file 格式
    '''
    
    def stackingTrainPort(self,data,label):
        super().stackingTrainPort(data,label)
    
    def stackingTestPort(self,data):
        return super().stackingTestPort(data)
    
    def stackingStage1FeatureAndLabel(self,data,label):
        probSet,labelSet = super().stackingStage1FeatureAndLabel(data,label)
        return probSet,labelSet
    
class rfMethod(standard_classification_template):
    def __init__(self):
        super(rfMethod, self).__init__()
        self.methodName = 'rf'
        #self.iterations = ''
        pass
    
    '''
    tree based model is meaningless to use daraNormalized, so we use origin data
    '''
    def dataNormalized(self,data):
        return data
        pass
    
    def establishModel(self):
        model = RandomForestClassifier(criterion='entropy',n_estimators=62 ,n_jobs=-1)
        return model
        pass
    
    '''
    为了增加可读性 需要再在子类添加一次接口
    '''
    def train(self,dataPath,labelPath):
        print('start rf train')
        super().train(dataPath, labelPath)
    
    def test(self,dataPath):
        print('start rf test')
        return super().test(dataPath)
    
    def KFoldEval(self,dataPath,labelPath):
        super().KFoldEval(dataPath,labelPath)
    
    '''
    stack 相关的三个接口输入是 numpy 格式的 数据 方便上级接口调用 而不是 file 格式的输入
    ''' 
    
    def stackingTrainPort(self,data,label):
        super().stackingTrainPort(data,label)
        
    def stackingTestPort(self,data):
        return super().stackingTestPort(data)

    def stackingStage1FeatureAndLabel(self,data,label):
        probSet,labelSet = super().stackingStage1FeatureAndLabel(data,label)
        return probSet,labelSet

class nbMethod(standard_classification_template):
    def __init__(self):
        super(nbMethod, self).__init__()
        self.methodName = 'nb'
        #self.iterations = ''
        pass
    
    def establishModel(self):
        model = GaussianNB()
        #model = MultinomialNB(alpha = 1.3, fit_prior=True)
        return model
        pass
    
    '''
    为了增加可读性 需要再在子类添加一次接口
    '''
    def train(self,dataPath,labelPath):
        print('start nb train')
        super().train(dataPath, labelPath)
    
    def test(self,dataPath):
        print('start nb test')
        return super().test(dataPath)
    
    def KFoldEval(self,dataPath,labelPath):
        super().KFoldEval(dataPath,labelPath)
    
    '''
    stack 相关的三个接口输入是 numpy 格式的 数据 方便上级接口调用 而不是 file 格式的输入
    ''' 
    def stackingTrainPort(self,data,label):
        super().stackingTrainPort(data,label)
    
    def stackingTestPort(self,data):
        return super().stackingTestPort(data)
    
    def stackingStage1FeatureAndLabel(self,data,label):
        probSet,labelSet = super().stackingStage1FeatureAndLabel(data,label)
        return probSet,labelSet
    
if __name__ == '__main__':
    dataPath = 'D:/2017_8_8_task/20170808/SAR/task_classification/comp_dataset/fc7.txt'
    labelPath = 'D:/2017_8_8_task/test/Stacking_test/labelinfo.txt'
    model = lrMethod()
    ''' 单元测试 - 1 '''
    model.train(dataPath,labelPath)
    model.test(dataPath)
    model.KFoldEval(dataPath,labelPath)
    print('pass the test stage 1')
    
    ''' stack 接口测试 '''
    #data = np.loadtxt(dataPath) 
    #label = np.loadtxt(labelPath)
    #model.stackingTrainPort(data,label)
    #result = model.stackingTestPort(data)
    #probSet,labelSet = model.stackingStage1FeatureAndLabel(data,label)
    #print('pass the test stage 2')