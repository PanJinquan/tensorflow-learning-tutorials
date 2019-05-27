import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

#模型路径
model_path = 'model/mnist.pb'
#测试数据
mnist = input_data.read_data_sets("Mnist_data/", one_hot=True)
x_test = mnist.test.images
x_labels = mnist.test.labels;

with tf.Graph().as_default():
    output_graph_def = tf.GraphDef()
    with open(model_path, "rb") as f:
        output_graph_def.ParseFromString(f.read())
        tf.import_graph_def(output_graph_def, name="")

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        # x_test = x_test.reshape(1, 28 * 28)
        input_x = sess.graph.get_tensor_by_name("input/x_input:0")
        output = sess.graph.get_tensor_by_name("output:0")

        # 【1】下面是进行批量测试----------------------------------------------------------
        pre_num = sess.run(output, feed_dict={input_x: x_test})#利用训练好的模型预测结果
        #结果批量测试的准确率
        correct_prediction = tf.equal(pre_num, tf.argmax(x_labels, 1,output_type='int32'))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        acc = sess.run(accuracy, feed_dict={input_x: x_test})
        # a = accuracy.eval({x: mnist.test.images, y_: mnist.test.labels})
        print('测试正确率：{0}'.format(acc))

        #【2】下面是进行单张图片的测试-----------------------------------------------------
        testImage=x_test[0];
        test_input = testImage.reshape(1, 28 * 28)

        pre_num = sess.run(output, feed_dict={input_x: test_input})#利用训练好的模型预测结果
        print('模型预测结果为：',pre_num)
        #显示测试的图片
        testImage = testImage.reshape(28, 28)
        testImage=np.array(testImage * 255, dtype="int32")
        fig = plt.figure(), plt.imshow(testImage, cmap='binary')  # 显示图片
        plt.title("prediction result:"+str(pre_num))
        plt.show()
        #保存测定的图片
        testImage = Image.fromarray(testImage)
        testImage = testImage.convert('L')
        testImage.save("data/test_image.jpg")
        # matplotlib.image.imsave('data/name.jpg', im)
        sess.close()


