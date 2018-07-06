import tensorflow as tf

tensor=tf.ones(shape=[2,1,3])
print("dtype:",tensor.dtype)
print("name:",tensor.name)
print("shape:",tensor.shape)
print("rank:",tensor.shape.ndims)
print("op:",tensor.op)
