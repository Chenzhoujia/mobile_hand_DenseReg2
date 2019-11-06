import tensorflow as tf

a = tf.constant([[[1,2],[3,4]],[[1,2],[3,4]]])# [f]
b = tf.constant([[[1,2],[3,4]],[[1,2],[-3,4]]])# [f]
ab = tf.matmul(a,b)
a_ = tf.transpose(a,[0,2,1])
print(tf.Session().run(ab))
print(tf.Session().run(a_))
