# DogVsCat
pytorch 使用AlexNet 实现猫狗二分类
——————————————————————————————————————————————————
## 1、数据格式

	data
	|--train
	|	|--cat
	|	|--dog
	|
	|--validation
		|--cat
		|--dog

## 2、文件

1-AlexNet.py为网络文件

2-train.py为训练文件，会在model文件夹下生成AlexNet.pth模型文件，经过70个epoch，准确率达到0.98左右，有过拟合的风险。


3-predict.py为预测文件
