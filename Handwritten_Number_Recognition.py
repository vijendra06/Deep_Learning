import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist= input_data.read_data_sets("/tmp/data/",one_hot= True)

nodesl1= 500
nodesl2= 500
nodesl3= 500
classes= 10
batch_size= 100

x= tf.placeholder('float' ,[None,784])
y= tf.placeholder('float')

def neuralnetworkmodel(data):
    hiddenl1= { 'weights' : tf.Variable(tf.random_normal([784, nodesl1])),
                'biases' : tf.Variable(tf.random_normal([nodesl1]))
               }
    hiddenl2= { 'weights' : tf.Variable(tf.random_normal([nodesl1, nodesl2])),
                'biases' : tf.Variable(tf.random_normal([nodesl2]))
               }
    hiddenl3= { 'weights' : tf.Variable(tf.random_normal([nodesl2, nodesl3])),
                'biases' : tf.Variable(tf.random_normal([nodesl3]))
               }
    outputlayer = { 'weights' : tf.Variable(tf.random_normal([nodesl3, classes])),
                'biases' : tf.Variable(tf.random_normal([classes]))
               }
    l1= tf.add(tf.matmul(data, hiddenl1['weights']), hiddenl1['biases'])
    l1= tf.nn.relu(l1)
    
    l2= tf.add(tf.matmul(l1, hiddenl2['weights']), hiddenl2['biases'])
    l2= tf.nn.relu(l2)
    
    l3= tf.add(tf.matmul(l2, hiddenl3['weights']), hiddenl3['biases'])
    l3= tf.nn.relu(l3)
    
    output= tf.matmul(l3, outputlayer['weights'])+ outputlayer['biases']
    
    
    return output

def trainneuralnetwork(x):
    prediction = neuralnetworkmodel(x)
    cost= tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer= tf.train.AdamOptimizer().minimize(cost)
    
    hmepochs= 10
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        
        for epoch in range(hmepochs):
            epochloss=0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epochx, epochy= mnist.train.next_batch(batch_size)
                _,c = sess.run([optimizer, cost], feed_dict={ x: epochx ,y: epochy})
                epochloss+=c
                
            print('epocccch: ',epoch,'   loss:',epochloss)
            
        correct= tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
        accuracy= tf.reduce_mean(tf.cast(correct, 'float'))
        print( 'Accuracy: ',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))
        
trainneuralnetwork(x)
            
    
    








