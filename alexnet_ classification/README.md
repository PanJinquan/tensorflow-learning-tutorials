# AlexNet_TF  
###### 利用Tensorflow简单实现AlexNet，从数据集制作到训练完成测试  
###### 参考《Tensorflow实战》和CSDN上相关BLOG
TFrecords格式数据制作参考:https://blog.csdn.net/sinat_16823063/article/details/53946549  
训练数据-17flowers，百度网盘链接: https://pan.baidu.com/s/1CXcCgC8Ch5Hdmkgde9yAww 密码: 3nc4  
* create_tfrecords.py为生成tfrecords数据脚本  
* alexnet.py为网络结构定义文件  
* alexnet_train.py为训练脚本  
* alexnet_test.py为测试脚本     

##### 制作tfrecord数据文件  
1. 下载17flowers数据集，解压到目录下  
```
    AlexNet
    |__ 17flowers
        	|__ 0
            	|__ xxx.JPEG
        	|__ 1
        		|__ xxx.JPEG
        	|__ 2
        		|__ xxx.JPEG
```
2. 执行create_tfrecords.py脚本，会在根目录下生成train.tfrecords文件，也可在脚本中指定生成路径    

##### 训练自己的数据  
1. 修改脚本中，模型保存位置及tfrecord数据所在路径，执行alexnet_train.py脚本即可训练  
2. 训练完成后生成模型文件，执行alexnet_test.py脚本即可进行测试
3. test文件夹中的图片名字前面数字即为所属类别





