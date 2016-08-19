#coding=utf-8
import data
import tensorflow as tf
CODE='0123456789abcdefghijklmnopqrstuvwxyz'

#model define
def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
def max_pool_2x2(x):
    return  tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
def max_pool_5x5(x):
    return tf.nn.max_pool(x,ksize=[1,5,5,1],strides=[1,1,1,1],padding='SAME')
sess=tf.Session()
x=tf.placeholder(tf.float32,[None,10000])
y1_=tf.placeholder(tf.float32,[None,36])
y2_=tf.placeholder(tf.float32,[None,36])
y3_=tf.placeholder(tf.float32,[None,36])
y4_=tf.placeholder(tf.float32,[None,36])
y5_=tf.placeholder(tf.float32,[None,36])
#first convolution layer
w_conv1=weight_variable([5,5,1,32])
b_conv1=bias_variable([32])
x_image=tf.reshape(x,[-1,100,100,1])
h_conv1=tf.nn.relu(conv2d(x_image,w_conv1)+b_conv1)
h_pool1=max_pool_2x2(h_conv1)
#second convolution layer
w_conv2=weight_variable([5,5,32,32])
b_conv2=bias_variable([32])
h_conv2=tf.nn.relu(conv2d(h_pool1,w_conv2)+b_conv2)
h_pool2=max_pool_2x2(h_conv2)
#dense layer
w_fc1=weight_variable([25*25*32,256])
b_fc1=bias_variable([256])
w_fc2 = weight_variable([25 * 25 * 32, 256])
b_fc2 = bias_variable([256])
w_fc3 = weight_variable([25 * 25 * 32, 256])
b_fc3 = bias_variable([256])
w_fc4 = weight_variable([25 * 25 * 32, 256])
b_fc4 = bias_variable([256])
w_fc5 = weight_variable([25 * 25 * 32, 256])
b_fc5 = bias_variable([256])


#h_poo12_flat=tf.reshape(h_pool2,[-1,7*7*32])
h_poo12_flat=tf.reshape(h_pool2,[-1,25*25*32])
h_fc1=  tf.nn.relu(tf.matmul(h_poo12_flat,w_fc1)  +b_fc1)
h_fc2 = tf.nn.relu(tf.matmul(h_poo12_flat, w_fc2) + b_fc2)
h_fc3 = tf.nn.relu(tf.matmul(h_poo12_flat, w_fc3) + b_fc3)
h_fc4 = tf.nn.relu(tf.matmul(h_poo12_flat, w_fc4) + b_fc4)
h_fc5 = tf.nn.relu(tf.matmul(h_poo12_flat, w_fc5) + b_fc5)

#dropout layer
keep_prob=tf.placeholder(tf.float32)
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)
h_fc2_drop=tf.nn.dropout(h_fc2,keep_prob)
h_fc3_drop=tf.nn.dropout(h_fc3,keep_prob)
h_fc4_drop=tf.nn.dropout(h_fc4,keep_prob)
h_fc5_drop=tf.nn.dropout(h_fc5,keep_prob)

#readout layer
w_fs=weight_variable([256,36])
b_fs=bias_variable([36])
y_conv1=tf.nn.softmax(tf.matmul(h_fc1_drop,w_fs)+b_fs)
y_conv2=tf.nn.softmax(tf.matmul(h_fc2_drop,w_fs)+b_fs)
y_conv3=tf.nn.softmax(tf.matmul(h_fc3_drop,w_fs)+b_fs)
y_conv4=tf.nn.softmax(tf.matmul(h_fc4_drop,w_fs)+b_fs)
y_conv5=tf.nn.softmax(tf.matmul(h_fc5_drop,w_fs)+b_fs)
#train and evaluate the model
cross_entropy1=tf.reduce_mean(-tf.reduce_sum(y1_*tf.log(tf.clip_by_value(y_conv1,1e-10,1.0)),reduction_indices=[1]))
cross_entropy2=tf.reduce_mean(-tf.reduce_sum(y2_*tf.log(tf.clip_by_value(y_conv2,1e-10,1.0)),reduction_indices=[1]))
cross_entropy3=tf.reduce_mean(-tf.reduce_sum(y3_*tf.log(tf.clip_by_value(y_conv3,1e-10,1.0)),reduction_indices=[1]))
cross_entropy4=tf.reduce_mean(-tf.reduce_sum(y4_*tf.log(tf.clip_by_value(y_conv4,1e-10,1.0)),reduction_indices=[1]))
cross_entropy5=tf.reduce_mean(-tf.reduce_sum(y5_*tf.log(tf.clip_by_value(y_conv5,1e-10,1.0)),reduction_indices=[1]))
cross_entropy=cross_entropy1+cross_entropy2+cross_entropy3+cross_entropy4+cross_entropy5

train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction1=tf.equal(tf.argmax(y_conv1,1),tf.argmax(y1_,1))
correct_prediction2=tf.equal(tf.argmax(y_conv2,1),tf.argmax(y2_,1))
correct_prediction3=tf.equal(tf.argmax(y_conv3,1),tf.argmax(y3_,1))
correct_prediction4=tf.equal(tf.argmax(y_conv4,1),tf.argmax(y4_,1))
correct_prediction5=tf.equal(tf.argmax(y_conv5,1),tf.argmax(y5_,1))
accuracy1=tf.reduce_mean(tf.cast(correct_prediction1,tf.float32))
accuracy2=tf.reduce_mean(tf.cast(correct_prediction2,tf.float32))
accuracy3=tf.reduce_mean(tf.cast(correct_prediction3,tf.float32))
accuracy4=tf.reduce_mean(tf.cast(correct_prediction4,tf.float32))
accuracy5=tf.reduce_mean(tf.cast(correct_prediction5,tf.float32))
accuracy=(accuracy1+accuracy2+accuracy3+accuracy4+accuracy5)/5;
saver=tf.train.Saver()


def tmain():
#if __name__=='__main__':
    saver.restore(sess,'/home/gpu/tensorflow/tensorflow/ai-cap/weight/weight6/model.ckpt')
    data2=data.DataSet()
    batch=data2.nextRandBatch()
    #train_accuracy=sess.run(accuracy,feed_dict={x:batch[0],y1_:batch[1][0],y2_:batch[1][1],y3_:
    #                                     batch[1][2],y4_:batch[1][3],y5_:batch[1][4],keep_prob:1.0})
    #print(train_accuracy)
    y1=sess.run(y_conv1,feed_dict={x:batch[0],y1_:batch[1][0],y2_:batch[1][1],y3_:
                                         batch[1][2],y4_:batch[1][3],y5_:batch[1][4],keep_prob:1.0})
    y2=sess.run(y_conv2,feed_dict={x:batch[0],y1_:batch[1][0],y2_:batch[1][1],y3_:
                                         batch[1][2],y4_:batch[1][3],y5_:batch[1][4],keep_prob:1.0})
    y3=sess.run(y_conv3,feed_dict={x:batch[0],y1_:batch[1][0],y2_:batch[1][1],y3_:
                                         batch[1][2],y4_:batch[1][3],y5_:batch[1][4],keep_prob:1.0})
    y4=sess.run(y_conv4,feed_dict={x:batch[0],y1_:batch[1][0],y2_:batch[1][1],y3_:
                                         batch[1][2],y4_:batch[1][3],y5_:batch[1][4],keep_prob:1.0})
    y5=sess.run(y_conv5,feed_dict={x:batch[0],y1_:batch[1][0],y2_:batch[1][1],y3_:
                                         batch[1][2],y4_:batch[1][3],y5_:batch[1][4],keep_prob:1.0})
    y1=list(y1[0]).index(max(y1[0]))
    y2=list(y2[0]).index(max(y2[0]))
    y3=list(y3[0]).index(max(y3[0]))
    y4=list(y4[0]).index(max(y4[0]))
    y5=list(y5[0]).index(max(y5[0]))
    print('y1:',CODE[y1],'y2:',CODE[y2],'y3:',CODE[y3],'y4:',CODE[y4],'y5:',CODE[y5])
    print CODE[y1]+CODE[y2]+CODE[y3]+CODE[y4]+CODE[y5]
    return CODE[y1]+CODE[y2]+CODE[y3]+CODE[y4]+CODE[y5]
