import tensorflow as tf

vec = tf.random_uniform(shape=(3,))
out1 = vec + 1
out2 = vec + 2
sess=tf.Session()
print(sess.run(out1))
print(sess.run(out2))
print(sess.run((out1,out2)))

