import tensorflow as tf
# to set up initial data
import numpy as np
# to visialize
import matplotlib.pyplot as plt

learning_rate = 0.01
training_epoch=50
# fake data
trX = np.linspace(-1,1,101)

num_coeffs = 6
trY_coeffs = [1,2,3,4,5,6]
trY = 0

for i in range(num_coeffs):
    trY += trY_coeffs[i] * np.power(trX,i)

trY += np.random.randn(*trX.shape) * 1.5

plt.scatter(trX,trY)
plt.show()

X = tf.placeholder("float")
Y = tf.placeholder("float")

# f(x)=wnxn +...+w2x2 +w1x+w0
def model(X,w):
    terms = []
    for i in range(num_coeffs):
        term = tf.mul(w[i],tf.pow(X,i))
        terms.append(term)
    return tf.add_n(terms)

w = tf.Variable([0.] * num_coeffs, name = "parameters")
y_model = model(X,w)

cost = (tf.pow(Y-y_model,2))
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for training_epoch in range(training_epoch):
    for(x,y) in zip(trX, trY):
        sess.run(train_op, feed_dict={X:x, Y:y})

w_val = sess.run(w)
print(w_val)

sess.close()

plt.scatter(trX,trY)
trY2=0
for i in range(num_coeffs):
    trY2 += w_val[i] * np.power(trX,i)
plt.plot(trX, trY2, 'r')
plt.show()

