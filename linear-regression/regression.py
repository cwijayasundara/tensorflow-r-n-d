import tensorflow as tf
# to set up initial data
import numpy as np
# to visialize
import matplotlib.pyplot as plt

# define hyper parameters
learning_rate = 0.01
training_epochs = 100

# set up the fake data which we need to find the best fit
x_train = np.linspace(-1,1,101)
y_train = 2 * x_train + np.random.randn(*x_train.shape) * 0.33

X = tf.placeholder("float")
Y = tf.placeholder("float")

#  define the model as y=w*x=wx
def model(X,w):
    return tf.mul(X,w)

# set the weights variable
w = tf.Variable(0.0, name="weights")

# def cost func
y_model = model(X,w)
cost = (tf.square(Y-y_model))

# define the func that will be called insie the loop
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for epoch in range(training_epochs):
    for(x,y) in zip(x_train,y_train):
        sess.run(train_op, feed_dict={X:x,Y:y})
        w_val = sess.run(w)

sess.close()
plt.scatter(x_train,y_train)
y_learned = x_train * w_val
plt.plot(x_train,y_learned,'r')
plt.show()
