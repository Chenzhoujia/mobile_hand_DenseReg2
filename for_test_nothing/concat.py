import tensorflow as tf

a = tf.constant([[[1,2,3,4],[5,6,7,8],[9,10,11,12]],[[13,14,15,16],[17,18,19,20],[21,22,23,24]]])# [f]
a_list  = []
for to_list in range(4):
    a_list.append(a[:,:,to_list])
a_ = tf.concat(a_list, axis=0)# [fA1,FA2,FB1,FB2,fC1,fC2,fd1,fd2]
a_shape = tf.reshape(a_,(4,2,3))#
a_list  = []
for to_list in range(4):
    a_list.append( tf.expand_dims(a_shape[to_list,:,:], -1))
a_list = tf.concat(a_list, axis=-1)# [fA1,FA2,FB1,FB2,fC1,fC2,fd1,fd2]
print(tf.Session().run(a))
print(tf.Session().run(a_))
print(tf.Session().run(a_shape))
print(tf.Session().run(a_list))