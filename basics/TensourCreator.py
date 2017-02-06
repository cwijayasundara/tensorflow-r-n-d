import tensorflow as tf

# 2*1 matrix
matrix1 = tf.constant([[1., 2.]])

# 1*2 matrix
matrix2 = tf.constant([[1],
                       [2]])

# define a rank = 3 matrix
matrix3 = tf.constant([[[1,2],
                        [3,4],
                        [5,6]],
                       [[7,8],
                        [9,10],
                        [11,12]]])

matrix4 = tf.ones([1000,1000])*0.2

neg_matrix1 = tf.neg(matrix1)

matrix5 =  tf.constant([[3., 4.]])

print(matrix1)
print (matrix2)
print (matrix3)
print (matrix4)
print (neg_matrix1)
print (tf.add(matrix1,matrix5))



