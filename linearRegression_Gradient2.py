# GradientDescent를 메소드로 구현
# b는 생략

import tensorflow as tf

X = [1, 2, 3]
Y = [1, 2, 3]

# 굉장히 어긋난 수 입력
# 컨벡스 그래프의 W값 생각
W = tf.Variable(5.0)

# Our hypothesis XW (b 생략)
hypothesis = X * W 
# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize: 제공되는 메소드로 구현
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)


# Launch the graph in a session
sess = tf.Session()
# Initializes global variables in the graph
sess.run(tf.global_variables_initializer())

for step in range(100):
    print(step, sess.run(W))
    sess.run(train)
