import tensorflow as tf
import numpy as np

# 2*2 matrix using plain python
m1 = [[1.0,2.0],
      [3.0,4.0]]
# 2*2 matrix using plain numpy
m2 = np.array([[1.0, 2.0],
               [3.0, 4.0]], dtype=np.float32)
# 2*2 matrix using plain tf
m3 = tf.constant([[1.0, 2.0],[3.0, 4.0]])

print(type(m1))
print(type(m2))
print(type(m3))

t1 = tf.convert_to_tensor(m1, dtype=tf.float32)
t2 = tf.convert_to_tensor(m2, dtype=tf.float32)
t3 = tf.convert_to_tensor(m3, dtype=tf.float32)

print(type(t1))
print(type(t2))
print(type(t3))