import tensorflow as tf
import numpy as np
from models import VGG16
import argparse
import scipy.misc
import os
from datetime import datetime as dt
import argparse
import time

# Add/Sub mean for image processing
mean = np.array([103.939, 116.779, 123.68], dtype=np.float32)
def add_mean(img):
    for i in range(3):
        img[0,:,:,i] += mean[i]
    return img

def sub_mean(img):
    for i in range(3):
        img[0,:,:,i] -= mean[i]
    return img

# doc anh
def read_img(path, w=None):
    img = scipy.misc.imread(path)
    # Chinh kich thuoc anh
    if w:
        r = w / np.float32(img.shape[1])
        img = scipy.misc.imresize(img, (int(img.shape[0]*r), int(img.shape[1]*r)))
    img = img.astype(np.float32)
    img = img[None, ...]
    
    img = sub_mean(img)
    return img

# Luu anh
def save_image(im, iteration, out_dir):
    img = im.copy()
    
    img = add_mean(img)
    img = np.clip(img[0, ...],0,255).astype(np.uint8)
    nowtime = dt.now().strftime('%H_%M_%S_%d_%m_%Y')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)    
    scipy.misc.imsave("{}/output_{}_iteration{}.png".format(out_dir, nowtime, iteration), img)
   
# Make command to run program 
def parseArgs():
    parser = argparse.ArgumentParser(
        description='A Neural Algorithm of Artistic Style')
    parser.add_argument('--modelpath', '-p', default='VGG_model',
                        help='Model file path')
    parser.add_argument('--content', '-cp', default='images/samsung.jpg',
                        help='Content image path')
    parser.add_argument('--style', '-sp', default='images/style.jpg',
                        help='Style image path')
    parser.add_argument('--iters', '-i', default=200, type=int,
                        help='Number of iterations')
    parser.add_argument('--alpha', '-a', default=1.0, type=float,
                        help='alpha (content weight)')
    parser.add_argument('--beta', '-b', default=200.0, type=float,
                        help='beta (style weight)')
    parser.add_argument('--device', default="/cpu:0")
    parser.add_argument('--out_dir', default="output")
    args = parser.parse_args()
    return args.content, args.style, args.modelpath, args.alpha, args.beta, args.iters, args.device, args

# Choose Convolutional model
def getModel(image, params_path):
    return VGG16(image, params_path)

# Main process

# Get value from run command 
content_img_path, style_img_path, params_path, alpha, beta, num_iters, device, args = parseArgs()
width = 600
# The actual calculation
print "Read images..."
content_img = read_img(content_img_path, width) #Anh noi dung
style_img   = read_img(style_img_path, width) #Anh style

g = tf.Graph()
with g.device(device), g.as_default(), tf.Session(graph=g, config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    print "Calculate convolutional layers for content image"
    image_content = tf.constant(content_img)    #Doc anh noi dung
    model_content = getModel(image_content, params_path)    # Lap anh noi dung vao model
    content_conv_layers = [sess.run(layer) for layer in model_content.layer_used()]  # Tinh cac feature maps tren cac lop Convolution

    print "Calculate convolutional layers for style image"
    image_style1 = tf.constant(style_img)  #Doc anh style 1
    model_style1 = getModel(image_style1, params_path)    # Lap anh style vao model
    layers = model_style1.layer_used()   #layers: tap cac lop convolution
    style_conv_layers = []
    for l in range(len(layers)): #Xet cac lop Convolution
        num_filters = content_conv_layers[l].shape[3] #Do sau (so filters) cua moi lop Convolution
        style_shape = [-1, num_filters]
        style_ = tf.reshape(layers[l], style_shape)
        style = tf.matmul(tf.transpose(style_), style_)  #Ma tran Gram cho style
        style_conv_layers.append(sess.run(style)) 

    # Bat dau tu anh nhieu den trang
    #gen_image = tf.Variable(tf.truncated_normal(content_img.shape, stddev=20), trainable=True, name='gen_image')
    # Bat dau tu anh content
    gen_image = tf.Variable(tf.constant(np.array(content_img, dtype=np.float32)), trainable=True, name='gen_image') #gen_image: anh can sinh
    model = getModel(gen_image, params_path)    # Lap anh can sinh vao model
    layers = model.layer_used()   #layers: tap cac lop Convolution cua anh can sinh
    Loss_content = 0.0
    Loss_style   = 0.0
    for l in range(len(layers)):
        # Content loss
        Loss_content += model.alpha[l]*tf.nn.l2_loss(layers[l] - content_conv_layers[l])
        # Style loss
        num_filters = content_conv_layers[l].shape[3]
        style_shape = [-1, num_filters]
        style_ = tf.reshape(layers[l], style_shape)
        style = tf.matmul(tf.transpose(style_), style_)  #Ma tran Gram cho gen_image
        N = np.prod(content_conv_layers[l].shape).astype(np.float32)
        Loss_style += model.beta[l]*tf.nn.l2_loss(style - style_conv_layers[l])/N**2/len(layers)    # Loss style 1
        
    L = alpha* Loss_content + beta * Loss_style

    # Optimizer
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(learning_rate=2.0, global_step=global_step, decay_steps=100, decay_rate=0.94, staircase=True)
    #train_step = tf.train.AdamOptimizer(learning_rate).minimize(L, global_step=global_step)
    
    #train_step = tf.train.AdagradOptimizer(learning_rate).minimize(L, global_step=global_step)

    train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(L, global_step=global_step)


    tf.scalar_summary("Loss_content", Loss_content)
    tf.scalar_summary("Loss_style", Loss_style)

    gen_image_addmean = tf.Variable(tf.constant(np.array(content_img, dtype=np.float32)), trainable=False)
    tf.image_summary("Generated image (TODO: add mean)", gen_image_addmean)
    summary_op = tf.merge_all_summaries()
    summary_writer = tf.train.SummaryWriter('/tmp/na-logs', graph_def=sess.graph_def)
    

    print "Calculate for genarate output image"

    sess.run(tf.initialize_all_variables())
    for i in range(num_iters):
        if i % 10 == 0: # moi 10 vong lap lai in ra loss va in ra anh
            generation_image = sess.run(gen_image)
            save_image(generation_image, i, args.out_dir)
            print "Loss_content: ", sess.run(Loss_content), "Loss_style: ", sess.run(Loss_style)

            sess.run(tf.assign(gen_image_addmean, add_mean(generation_image)))
            summary_str = sess.run(summary_op)
            summary_writer.add_summary(summary_str, i)
        print "Iteration:", i
        sess.run(train_step)

