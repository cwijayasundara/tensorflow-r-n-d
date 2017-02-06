# my first Tensorflow app
import tensorflow as tf

matrix = tf.constant([[1.,2.]])
negMatrix = tf.neg(matrix)

with tf.Session() as sess:
    result = sess.run(negMatrix)
print (result)