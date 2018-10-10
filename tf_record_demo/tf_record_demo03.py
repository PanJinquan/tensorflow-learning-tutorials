# -----------------------------------------------------------------------------
# encoding=utf-8
import os
import tensorflow as tf
from PIL import Image

cwd = './test_image'
classes = {'dog', 'cat'}


# 制作TFRecords数据
def create_record():
    writer = tf.python_io.TFRecordWriter("dog_train.tfrecords")
    for index, name in enumerate(classes):
        class_path = cwd + "/" + name + "/"
        for img_name in os.listdir(class_path):
            img_path = class_path + img_name
            img = Image.open(img_path)
            img = img.resize((64, 64))
            img_raw = img.tobytes()  # 将图片转化为原生bytes
            print(index, img_raw)
            example = tf.train.Example(
                features=tf.train.Features(feature={
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                    'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                }))
            writer.write(example.SerializeToString())
    writer.close()


# -------------------------------------------------------------------------

# 读取二进制数据

def read_and_decode(filename):
    # 创建文件队列,不限读取的数量
    filename_queue = tf.train.string_input_producer([filename])
    # create a reader from file queue
    reader = tf.TFRecordReader()
    # reader从文件队列中读入一个序列化的样本
    _, serialized_example = reader.read(filename_queue)
    # get feature from serialized example
    # 解析符号化的样本
    features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'img_raw': tf.FixedLenFeature([], tf.string)
        })
    label = features['label']
    img = features['img_raw']
    img = tf.decode_raw(img, tf.uint8)
    img = tf.reshape(img, [64, 64, 3])
    # img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(label, tf.int32)
    return img, label


# ---------主程序----------------------------------------------------------
if __name__ == '__main__':
    create_record()
    batch = read_and_decode('dog_train.tfrecords')
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:  # 开始一个会话
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(10):
            example, lab = sess.run(batch)  # 在会话中取出image和label
            img = Image.fromarray(example, 'RGB')  # 这里Image是之前提到的
            img.save(cwd + '/val/' + str(i) + '_Label_' + str(lab) + '.jpg')  # 存下图片;注意cwd后边加上‘/’
            print(example, lab)
        coord.request_stop()
        coord.join(threads)
        sess.close()
# -----------------------------------------------------------------------------
