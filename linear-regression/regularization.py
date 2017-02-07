import tensorflow as tf
# to set up initial data
import numpy as np
# to visialize
import matplotlib.pyplot as plt

def split_dataset(x_dataset,y_dataset,ratio):
    arr = np.arange(x_dataset.size)
    np.random.shuffle(arr)
    num_train = ratio * x_dataset.size
    x_train = x_dataset[arr[0:int(num_train)]]
    y_train = y_dataset[arr[0:int(num_train)]]
    x_test = x_dataset[arr[int(num_train):x_dataset.size]]
    y_test = y_dataset[arr[int(num_train):x_dataset.size]]
    return x_train, x_test,y_train,y_test

learning_rate = 0.001
training_epochs = 1000
reg_lamdba = 0.

x_dataset = np.linspace(-1,1,100)
# to set up initial data y=x*x
num_coeffs = 9
y_dataset_params = [0.] * num_coeffs
y_dataset_params[2] = 1
y_dataset = 0

for i in range(num_coeffs):
    y_dataset += y_dataset_params[i] * np.power(x_dataset,i)
    y_dataset += np.random.randn(*x_dataset.shape) * 0.3

plt.scatter(x_dataset,y_dataset)
plt.show()

(x_train, x_test, y_train, y_test) = split_dataset(x_dataset,y_dataset,0.7)

X = tf.placeholder("float")
Y = tf.placeholder("float")

def model(X,w):
    terms = []
    for i in range(num_coeffs):
        term = tf.mul(w[i],tf.pow(X,i))
        terms.append(term)
        return tf.add_n(terms)

w = tf.Variable([0.] * num_coeffs, name = "parameters")
y_model = model(X,w)

cost = tf.div(tf.add(tf.reduce_sum(tf.square(Y-y_model)), tf.mul(reg_lamdba,tf.reduce_sum(tf.square(w)))),2*x_train.size)

train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for reg_lamdba in np.linspace(0,1,100):
    for epoch in range(training_epochs):
        sess.run(train_op, feed_dict={X:x_train, Y: y_train})
    final_cost=sess.run(cost,feed_dict={X: x_test, Y: y_test})
    print ('reg lambda', reg_lamdba)
    print ('final cost', final_cost)

sess.close()
