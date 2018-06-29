#导入tensorflow库
import tensorflow  as tf
#输入出tf版本号
print(tf.__version__)
#构建一个字符串常量
str = tf.constant('Hello, TensorFlow!')
#构建TensorFlow运行会话
sess = tf.Session()
#运行会话，输出字符串常量
print(sess.run(str))

