import pandas as pd
trainb = pd.read_csv("E:\\paper\\trainb.txt",sep="\t")
trainby = pd.read_csv("E:\\paper\\trainby.txt",sep="\t")
testb = pd.read_csv("E:\\paper\\testb.txt",sep="\t")
testby = pd.read_csv("E:\\paper\\testby.txt",sep="\t")

import tensorflow as tf

x = tf.placeholder(tf.float32, [None, 2183])
y_ = tf.placeholder(tf.float32, [None, 2])

w1 = tf.Variable(tf.random_normal([2183, 256]))
b1 = tf.Variable(tf.random_normal([256]))
z1 = tf.add(tf.matmul(x, w1), b1)
h1 = tf.nn.softmax(z1)

w0 = tf.Variable(tf.random_normal([256, 2]))
b0 = tf.Variable(tf.random_normal([2]))
logits = tf.add(tf.matmul(h1, w0), b0)
y = tf.nn.softmax(logits)

predict = tf.argmax(y,1)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(3).minimize(cross_entropy)
correct_prediction = tf.equal(predict, tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()
sess = tf.Session()

sess.run(init)

for i in range(1501):
	sess.run(train_step, feed_dict={x: trainb, y_: trainby})
	if i % 20 == 0 :
		print(sess.run([cross_entropy, accuracy], feed_dict={ x: trainb, y_: trainby }))

print(sess.run(accuracy, feed_dict={ x: testb, y_: testby}))
# print(sess.run(auc, feed_dict={ x: testb, y_: testby}))
prediction = sess.run(predict, feed_dict={ x: testb, y_: testby})
df=pd.DataFrame(prediction)
df.to_csv ('E:\\paper\\prediction.csv', index = None, header=False)