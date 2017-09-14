# 级联 stacking 森林

# 模板支持 
* 标准模板 ： 支持 scikit-learn style 的算法部分 包含 SVM LR NB RF 等
* 额外模板 ： 支持 xgboost 未来会添加 lightgbm fm 等算法 

# 使用说明 ：
              standard_template,py / external_template.py : 支持 train test 和 K 折交叉验证接口   数据格式目前支持 txt 文档 未来会添加 xml json 等

	      cascade forest.py : 支持 stacking 算法 ，支持任意层数 , 支持自定义集成函数


