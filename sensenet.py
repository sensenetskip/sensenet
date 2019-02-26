import tensorflow as tf
import numpy as np
import os
import sys
import time
from datetime import timedelta
import shutil
import cv2
sys.path.append('../data_provider/')
import camviddata as cam
TF_VERSION = float('.'.join(tf.__version__.split('.')[:2]))


class rnet:
    def __init__(self, data_provider=None):
        self.data_shape = data_provider.data_shape
        self.n_classes = data_provider.n_classes
        self.growth_rate = 12
        self.layers_per_block = 5
        self.first_output_features = 24
        self.keep_prob = 0.8
        self.data_provider = data_provider
        self.weight_decay = 1e-4
        tf.reset_default_graph()
        self._define_inputs()
        self._build_graph()
        self._initialize_session()
        self._count_trainable_params()
        self.batch_size = 2
        # self.layeroutputs = []
        print("label size received" + str(data_provider.train.labels[0].shape))

    def _define_inputs(self):
        # define the inputs
        shape = [None]
        shape.extend(self.data_shape)
        # print("@@@@@"+str(shape))
        labelshape = [None]
        labelshape.extend((self.data_shape[0], self.data_shape[1], 12))
        print("$$$" + str(labelshape))

        self.images = tf.placeholder(
            tf.float32,
            shape=shape,
            name='input_images')
        print(self.images.shape)

        self.labels = tf.placeholder(
            tf.float32,
            shape=labelshape,
            name='label_images')

        self.learning_rate = tf.placeholder(
            tf.float32,
            shape=[],
            name='learning_rate')

        self.is_training = tf.placeholder(tf.bool, shape=[])

    def _initialize_session(self):
        # initialize the session and variables of the graph

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        self.sess.run(tf.global_variables_initializer())

        logswriter = tf.summary.FileWriter
        self.saver = tf.train.Saver()
        self.summary_writer = logswriter("./log/")

    def one_hot_it(self, labels, w, h):
        # print("labels in one hot"+str(labels.shape))
        batch_size = labels.shape[0]
        x = np.zeros([batch_size, w, h, 12])
        for k in range(batch_size):
            for i in range(w):
                for j in range(h):
                    x[batch_size, i, j, labels[0, i, j]] = 1
        return x

    def _count_trainable_params(self):
        total_parameters = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_parametes = 1
            for dim in shape:
                variable_parametes *= dim.value
            total_parameters += variable_parametes
        print("Total training params: %.1fM" % (total_parameters / 1e6))

    @property
    def save_path(self):
        pass

    @property
    def logs_path(self):
        try:
            logs_path = self._logs_path
        except AttributeError:
            logs_path = 'logs/%s' % self.model_identifier
            if self.renew_logs:
                shutil.rmtree(logs_path, ignore_errors=True)
            os.makedirs(logs_path)
            self._logs_path = logs_path
        return logs_path

    @property
    def model_identifier(self):
        pass

    def save_model(self, global_step=None):
        self.saver.save(self.sess, './', global_step=global_step)

    def load_model(self):
        save_path = './modelname'
        save_path = os.path.join(save_path, 'model.chkpt')
        self._save_path = save_path
        print('model path is %s' %save_path)
        try:
            self.saver.restore(self.sess, save_path )
        except Exception as e:
            raise IOError("Failed to to load model from save path: %s" % self.save_path)
        self.saver.restore(self.sess, save_path)
        print("Successfully load model from save path: %s" % self.save_path)

    def log_loss_accuracy(self, loss, accuracy, t_err, epoch, prefix):
        print("mean cross_entropy: %f, mean accuracy: %f, error: %f" % (
                loss, accuracy, t_err))
        summary = tf.Summary(value=[
            tf.Summary.Value(
                tag='loss_%s' % prefix, simple_value=float(loss)),
            tf.Summary.Value(
                tag='accuracy_%s' % prefix, simple_value=float(accuracy)),
            tf.Summary.Value(
                tag='error_%s' % prefix, simple_value=float(t_err))
        ])
        self.summary_writer.add_summary(summary, epoch)


    def composite_function(self, _input, out_features, kernel_size=3,
                           padding="SAME", strides=[1, 1, 1, 1]):
        """Function from paper H_l that performs:
        - batch normalization
        - ReLU nonlinearity
        - convolution with required kernel
        - dropout, if required
        """
        with tf.variable_scope("composite_function"):
            # BN
            output = self.batch_norm(_input)
            # ReLU
            output = tf.nn.relu(output)
            # convolution
            output = self.conv2d(output, out_features=out_features,
                                 kernel_size=kernel_size, padding=padding,
                                 strides=strides)
            # dropout(in case of training and in case it is no 1.0)
            output = self.dropout(output)
        return output

    def conv2d(self, _input, out_features, kernel_size,
               strides=[1, 1, 1, 1], padding='SAME'):
        in_features = int(_input.get_shape()[-1])
        kernel = self.weight_variable_xavier(
            [kernel_size, kernel_size, in_features, out_features],
            name='kernel')
        output = tf.nn.conv2d(_input, kernel, strides, padding)
        return output

    def max_pool(self, _input, k):
        ksize = [1, k, k, 1]
        strides = [1, k, k, 1]
        padding = 'VALID'
        output = tf.nn.max_pool(_input, ksize, strides, padding)
        return output

    def batch_norm(self, _input):
        output = tf.contrib.layers.batch_norm(
            _input, scale=True, is_training=self.is_training,
            updates_collections=None, epsilon=0.0010, decay=0.9, center=True)
        return output

    def dropout(self, _input):
        if self.keep_prob < 1:
            output = tf.cond(
                self.is_training,
                lambda: tf.nn.dropout(_input, self.keep_prob),
                lambda: _input
            )
        else:
            output = _input
        return output

    def bottleneck(self, _input, out_features, final=False):
        with tf.variable_scope("bottleneck"):
            output = self.batch_norm(_input)
            output = tf.nn.relu(output)
            if final:
                inter_features = out_features
            else:
                inter_features = out_features * 4
            output = self.conv2d(
                output, out_features=inter_features, kernel_size=1,
                padding='VALID')
            output = self.dropout(output)
        return output

    def layer(self, _input, growth_rate):
        bottleneck_out = self.bottleneck(_input, out_features=growth_rate)

        comp_out = self.composite_function(bottleneck_out,
                                           out_features=growth_rate,
                                           kernel_size=3)
        if TF_VERSION >= 1.0:
                output = tf.concat(axis=3, values=(_input, comp_out))
        else:
            output = tf.concat(3, (_input, comp_out))
        return output

    def weight_variable_heuniform(self, shape, name):
        return tf.get_variable(
            name=name,
            shape=shape,
            initializer=tf.contrib.layers.he_initializer())

    def weight_variable_msra(self, shape, name):
        return tf.get_variable(
            name=name,
            shape=shape,
            initializer=tf.contrib.layers.variance_scaling_initializer(
                mode="FAN_AVG"))

    def weight_variable_xavier(self, shape, name):
            return tf.get_variable(name, shape=shape,
                                   initializer=tf.contrib.layers.
                                   xavier_initializer(uniform=False))

    def bias_variable(self, shape, name='bias'):
        initial = tf.constant(0.0, shape=shape)
        return tf.get_variable(name, initializer=initial)

    def add_block(self, _input, growth_rate, layers_per_block):
        """Add N H_l internal layers"""
        output_features = _input
        for layer in range(layers_per_block):
            with tf.variable_scope("layer_%d" % layer):
                output_features = self.layer(output_features, growth_rate)

        # output_features = int(int(output_features.get_shape()[-1]) * self.reduction)
        print("convolution"+str(output_features.shape))
        return output_features

    def add_dblock(self, _input, growth_rate, layers_per_block):
        """Add N H_l internal layers"""
        output_features = _input
        print("input to add deconv block"+str(_input.shape))
        for layer in range(layers_per_block):
            with tf.variable_scope("layer_%d" % layer):
                output_features = self.layer(output_features, growth_rate)

        # output_features = int(int(output_features.get_shape()[-1]) * self.reduction)
        
        output_features = output_features[:,:,:,_input.shape[-1]:]
        print("deconvolution2"+str(output_features.shape))
        return output_features

    def transition_layer(self, _input, out_features):
        """Call H_l composite function with 1x1 kernel and after average
        pooling
        """
        # call composite function with 1x1 kernel
        # out_features = int(int(_input.get_shape()[-1]) * self.reduction)

        print("input of transition layer:  %d", str(_input.shape[-1]))

        output = self.composite_function(_input, out_features=out_features,
                                         kernel_size=1, padding="VALID",
                                         strides=[1, 1, 1, 1])
        # run average pooling
        output = self.max_pool(output, k=2)

        print("Transition layer input:"+str(_input.shape[-1])+"out_features: "+str(out_features)+"after pooling:"+str(output.shape))
        return output

    def upconv_concat(self, _input, residual):
        print("inptut of up transition is: %d", str(_input.shape[-1]))
        with tf.variable_scope("comp_function"):
            out = self.batch_norm(_input)
            out = tf.nn.relu(out)
            outputshape = residual.get_shape()
            print("input " + str(_input.shape))
            print("residual " + str(residual.shape))
            output = self.upconv_2d(out, outputshape)
            print("output " + str(output.shape))
        return tf.concat(axis=3, values=(residual,output))

    def upconv_2d(self, _input, outputshape):

        in_channels = outputshape[-1]

        in_features = int(_input.get_shape()[-1])

        kernel = self.weight_variable_xavier(
            [3, 3, in_features, in_features],
            name='kernel')

        dyn_input_shape = _input.get_shape()
        batch_s = dyn_input_shape[0]
        x_shape = tf.shape(_input)[0]
        out_shape = tf.stack([x_shape, dyn_input_shape[1] * 2,
                             dyn_input_shape[2] * 2, in_features])
        # kernel = [2, 2, 32, in_channels]
        # out_shape = tf.stack([32,outputshape[1],outputshape[2],32])
        strides = [1, 2, 2, 1]
        with tf.variable_scope("deconv"):
            output = tf.nn.conv2d_transpose(_input, kernel,
                                            out_shape, strides, padding='SAME')

        output = tf.reshape(output, out_shape)
        return output

    def _build_graph(self):
        growth_rate = self.growth_rate
        layers_per_block = self.layers_per_block

        keep_prob = tf.placeholder(tf.float32)
        downConvolutions = []

        with tf.variable_scope("Initial_convolution"):
            output = self.conv2d(
                self.images,
                out_features=self.first_output_features,
                kernel_size=3)
        print('Initial convolution size'+str(output.shape))

        for i in range(0, 4):
            for k in range(0, 1):
                with tf.variable_scope("block_" + str(i) + "_" + str(k)):
                    output = self.add_block(output, self.growth_rate,
                                            self.layers_per_block)
                print('block in denseblock '+str(output.shape))    
            downConvolutions.append(output[:, :, :, (output.get_shape()[-1] -
                                    self.growth_rate):])
            # downConvolutions.append(output)
            # if i != 3 - 1:
            with tf.variable_scope("Transition_after_block_%d" % i):
                output = self.transition_layer(output, output.shape[-1])

                print('output m after DB+TD = ' + str(i) + " "+str(output.shape[-1]))
        with tf.variable_scope("mid_block"):
            output = self.add_block(output, self.growth_rate, 3)
        print('output m after midblock = ' + str(output.shape[-1]))

        for j in range(3, -1, -1):
            with tf.variable_scope("block_deconv%d" % j):
                outputshape = output.get_shape()[-1]
                print("outputshape"+str(outputshape))
                output = self.upconv_concat(output, downConvolutions[j])
                print("!!!!" + str(output.shape))
                # output = tf.reshape(output, [-1, output.shape[1], output.shape[2], output.shape[3]])
                with tf.variable_scope("conv%d" % j):
                    output = self.add_dblock(output, self.growth_rate,
                                            self.layers_per_block)

            print('output after TU+DB' + str(j)+str(output.shape[-1]))
        # output = self.images
        with tf.variable_scope("final_conv"):
            print("input to final is: " + str(output.shape[-1]))
            output = self.bottleneck(output, 12, final=True)
            #bias = self.bias_variable([12])
            #output = tf.nn.bias_add(output, bias)
            print("final_shape " + str(output.shape))
            # output = tf.nn.relu(self.batch_norm(output))
        #output = tf.nn.softmax(output)
        #prediction = tf.nn.softmax(output)
        # flat_output = tf.reshape(tensor=output[:,:,:,0:11],shape=(-1,11))
        # flat_label = tf.reshape(tensor=self.labels[:,:,:,0:11],shape=(-1,11))
        self.layeroutputs = []
        self.origoutputs = []
        self.layeroutputs.append(output)
        self.origoutputs.append(self.labels)
        class_weights = tf.constant([[0.026070907004117, 0.022608056770053, 0.207985660992568,
                          0.019937425874446, 0.052188913862554, 0.034111367373255, 
                          0.154031357802302, 0.111921100089368, 0.044826639660709,
                          0.143370953519315,0.182947617051313, 0.0001]])
        
        class_weights = tf.constant([[0.58872014284134, 0.51052379608154, 4.6966278553009,
                          0.45021694898605, 1.1785038709641, 0.77028578519821, 
                          3.4782588481903, 2.5273461341858, 1.0122526884079,
                          3.2375309467316, 4.1312313079834, 0]])
       
        #class_weights = [0.2595, 0.1826, 4.5640,
        # 0.1417, 0.5051, 0.3826,
        # 9.6446, 1.8418, 6.6823,
        # 6.2478, 3.0, 7.3614]

        #weights = tf.gather(class_weights, tf.cast(self.labels, tf.int32))
        weights = tf.reduce_sum(class_weights * self.labels, axis=3)
        # weights = class_weights
        # self.loss = -tf.losses.sparse_softmax_cross_entropy(logits=output,labels=tf.cast(self.labels,tf.int32),weights=weights)
        unweighted_loss = tf.nn.softmax_cross_entropy_with_logits(labels=tf.cast(self.labels,tf.int32), logits=output)
        #self.loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels = self.labels, logits = output))
        #print('weights'+str(class_weights.shape) + ' ' + str(weights.shape) + ' ' + str(unweighted_loss.shape))
        weighted_loss = unweighted_loss * weights
        print('weighted_loss shape',str(weighted_loss.shape))

        self.loss = tf.reduce_mean(weighted_loss)
        # self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=flat_output,labels=flat_label))
        # self.loss = -tf.losses.log_loss(self.labels, output, weights=weights)
        self.iou = tf.reduce_mean(self.IOU2_(y_pred=output,
                                    y_true=self.labels))
        #self.loss = self.IOU_(y_pred=output, y_true=self.labels)
        self.cost = tf.reduce_mean(self.loss)
        l2_loss = tf.add_n(
            [tf.nn.l2_loss(var) for var in tf.trainable_variables()])

        # self.loss = tf.nn.softmax(logits=output)
        #global_step = tf.train.get_or_create_global_step()
        #optim = tf.train.MomentumOptimizer(self.learning_rate, 0.009,
        #                                   use_nesterov=True)
        optim = tf.train.RMSPropOptimizer(self.learning_rate,decay=0.0000001)
        self.train_step = optim.minimize(self.cost  *
                                         self.weight_decay
                                         )
        self.accuracy = self.get_predictions(output,self.labels)

    def IOU2_(self, y_pred, y_true):
        """Returns a (approx) IOU score
        intesection = y_pred.flatten() * y_true.flatten()
        Then, IOU = 2 * intersection / (y_pred.sum() + y_true.sum() + 1e-7) + 1e-7
        Args:
        y_pred (4-D array): (N, H, W, 1)
        y_true (4-D array): (N, H, W, 1)
        Returns:
        float: IOU score
        """
        #class_weights = np.array([0.02595, 8.0826, 4.5640, 0.1417,
        #                          0.5051, 0.3826, 9.6446, 1.8418,
        #                          6.6823, 6.2478, 3.0, 7.3614])
        # weights = tf.gather(class_weights, tf.cast(self.labels,tf.int32))

        print('pred:' + str(y_pred.shape) + ' gt: ' + str(y_true.shape))
        totalvalue = []
        nom = []
        denom = []
        for i in range(0, 11):
            y2_pred = y_pred[:, :, :, i]
            y2_pred = tf.reshape(y2_pred, [-1, 224, 224, 1])
            y2_true = y_true[:, :, :, i]
            y2_true = tf.reshape(y2_true, [-1, 224, 224, 1])
            H, W, C = y2_pred.get_shape().as_list()[1:]
            pred_flat = tf.reshape(y2_pred, [-1, H * W * C])
            true_flat = tf.reshape(y2_true, [-1, H * W * C])
            intersection = tf.reduce_sum(pred_flat * true_flat, axis=1)
            print(type(totalvalue))
            denominator = tf.subtract((tf.reduce_sum(
                                       pred_flat, axis=1) + tf.reduce_sum(
                                       true_flat, axis=1)), intersection)
            # totalvalue.append((intersection / denominator))
            nom.append(intersection)
            denom.append(denominator)
        # data_np = np.asarray(totalvalue, np.float32)
        # data_np = tf.stack([totalvalue])
        n = tf.reduce_sum(tf.stack([nom]))
        d = tf.reduce_sum(tf.stack([denom]))
        # val = tf.reduce_sum(data_np)
        val = tf.divide(n, d)
        return tf.reduce_sum(val)

    def IOU_(self, y_pred, y_true):

        """Returns a (approx) IOU score
        intesection = y_pred.flatten() * y_true.flatten()
        Then, IOU = 2 * intersection / (y_pred.sum() + y_true.sum() + 1e-7) + 1e-7
        Args:
           y_pred (4-D array): (N, H, W, 1)
           y_true (4-D array): (N, H, W, 1)
       Returns:
           float: IOU score
       """
        print('pred:' + str(y_pred.shape) + ' gt: ' + str(y_true.shape))
        y_pred = y_pred[:, :, :, 0:10]
        y_true = y_true[:, :, :, 0:10]
        H, W, C = y_pred.get_shape().as_list()[1:]
        pred_flat = tf.reshape(y_pred, [-1, H * W * C])
        true_flat = tf.reshape(y_true, [-1, H * W * C])
        intersection = tf.reduce_sum(pred_flat * true_flat, axis=1)
        denominator = tf.subtract((tf.reduce_sum(
                                   pred_flat, axis=1) + tf.reduce_sum(
                                   true_flat, axis=1)), intersection)
        return (intersection / denominator)

    def train_all_epochs(self, n_epochs):
        # n_epochs = train_params['n_epochs']
        # learning_rate = train_params['initial_learning_rate']
        # batch_size = train_params['batch_size']
        reduce_lr_epoch_1 = 150

        reduce_lr_epoch_2 = 225
        total_start_time = time.time()
        allepochacc = []
        learning_rate = 0.01
        # batch = self.data_provider.train.next_batch(1)
        for epoch in range(1, n_epochs + 1):
            print("\n", '-' * 30, "Train epoch: %d" % epoch, '-' * 30, '\n')
            start_time = time.time()
            # if epoch == reduce_lr_epoch_1:
            #    learning_rate = learning_rate / 5
            #    print("Decrease learning rate, new lr = %f" % learning_rate)
            # if epoch == reduce_lr_epoch_2:
            #    learning_rate = learning_rate / 5
            #    print("Decrease learning rate, new lr = %f" % learning_rate)
            # if epoch % 10 == 0:
            

            print("Training...")
            loss, iou,acc  = self.train_one_epoch(epoch,
                                        self.data_provider.train,
                                        self.batch_size, learning_rate)
            #if epoch % 30 == 0:
            #learning_rate = learning_rate * (0.995 ** (epoch/600))
            initial_learning_rate = 0.0001
            dr = 0.00001
            #learning_rate = initial_learning_rate  * (dr ** ((epoch+1)/10.0))
            print('decreasing learning rate = '+str(learning_rate))

            print(str(loss) + ' iou '+str(iou)+ ' Average Accuracy '+str(acc)+' Average Error '+str(1-acc)+'\n')
            self.log_loss_accuracy(loss, acc, iou , epoch, prefix='train')

            
            # if self.should_save_logs:
            #    self.log_loss_accuracy(loss, acc, err, epoch, prefix='train')

            # if train_params.get('validation_set', False):
            print("Validation...")
            loss, err,acc  = self.test(self.data_provider.validation, self.batch_size)
            #    allepochacc.append(acc)
            #    if self.should_save_logs:
            self.log_loss_accuracy(loss, acc, err, epoch, prefix='valid')

            time_per_epoch = time.time() - start_time
            seconds_left = int((n_epochs - epoch) * time_per_epoch)
            print("Time per epoch: %s, Est. complete in: %s" % (
                str(timedelta(seconds=time_per_epoch)),
                str(timedelta(seconds=seconds_left))))

            # if self.should_save_model:
            self.save_model()

        total_training_time = time.time() - total_start_time
        print("\nTotal training time: %s" % str(timedelta(
            seconds=total_training_time)))
        # print("\nBestValidationError =  %f" %(100-np.max(allepochacc)))
    def get_predictions(self,output_batch,labels):
        batch, w, h, c = output_batch.get_shape()
        
        n_pixels = batch * w * h
        correct_prediction = tf.equal(tf.argmax(output_batch[:,:,:,0:10], -1), tf.argmax(labels[:,:,:,0:10], -1))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return accuracy

        



    def train_one_epoch(self, epoch, data, batch_size,
                        learning_rate, batch=None):
        num_examples = data.num_examples
        total_loss = []
        total_accuracy = []
        total_err = []
        accSum = 0.0

        for i in range(num_examples // batch_size):
        # for i in range(1):
            batch = data.next_batch(batch_size)
            
            images, labels = batch
            

            # result = np.zeros_like(labels)

            # print('adsfasdfasdfasdfasdfasdfasdfasdfasdfasdfasdf'+str(labels.shape))
            # lab = self.one_hot_it(labels,224,224)
            # test = lab[0,:,:,:]
            # result = np.zeros_like(test)
            # print('adsfasdfasdfasdfasdfasdfasdfasdfasdfasdfasdf'+str(labels.shape))
            # for i in range(0,11):
            #    result[:,:,i] = cv2.medianBlur(test[:,:,i],5)
            # lab[0,:,:,:] = result
            num_samples = labels.shape[0]
            feed_dict = {
                self.images: images,
                self.labels: labels,
                self.learning_rate: learning_rate,
                self.is_training: True,
            }
            fetches = [self.train_step, self.cost,
                       self.layeroutputs, self.origoutputs, self.iou, self.accuracy]
            d, result, layeroutputs, origoutputs, iou,acc = self.sess.run(fetches,
                                                                 feed_dict=feed_dict)
            t_error = result
            print(str(t_error) + ' iou '+ str(iou))
            #self.log_loss_accuracy(loss, acc, err, epoch*i+i, prefix='train_all')
            if i % 10 == 0:
                with open('data.npy', 'w') as fid:
                    np.save(fid, np.array(layeroutputs))
                with open('label.npy', 'w') as fid:
                    np.save(fid, np.array(origoutputs))
            # print("error received"+str(np.mean(result))+" and d"+str(d))
            # total_loss.append(loss)
            total_accuracy.append(acc)
            # accSum = accSum + accuracy*num_samples
            # print("epoch %f training Error: %7.3f" %(epoch, 100.0-accuracy))

            # N=N+num_samples
            # total_err.append(t_error)
            # if self.should_save_logs:
            #    self.batches_step += 1
            #    self.log_loss_accuracy(
            #        loss, accuracy, t_error, self.batches_step, prefix='per_batch',
            #        should_print=False)
        # mean_loss = np.mean(total_loss)
        mean_accuracy = np.mean(total_accuracy)
        # mean_accuracy = accSum/N
        # mean_error = np.mean(total_err)
        # return mean_loss, mean_accuracy, mean_error
        return t_error, iou, mean_accuracy

    def test(self, data, batch_size,epoch=-1):
        num_examples = data.num_examples
        total_loss = []
        total_accuracy = []
        total_err = []
        accSum = 0.0
        N = 0
        for i in range(num_examples // batch_size):
            batch = data.next_batch(batch_size)
            num_samples = batch[0].shape[0]
            feed_dict = {
                self.images: batch[0],
                self.labels: batch[1],
                self.is_training: False,
            }
            fetches = [ self.cost,
                       self.layeroutputs, self.origoutputs, self.iou, self.accuracy]
            result, layeroutputs, origoutputs, iou, acc = self.sess.run(fetches, feed_dict=feed_dict)
            if i % 10 == 0:
                with open('valdata.npy', 'w') as fid:
                    np.save(fid, np.array(layeroutputs))
                with open('vallabel.npy', 'w') as fid:
                    np.save(fid, np.array(origoutputs))
            #total_loss.append(result * batch_size)
            #N = N + num_samples
            #accSum = accSum + accuracy * num_samples
            total_accuracy.append(acc)
            if epoch == -1:
                print("epoch %f validation Error: %7.3f (%7.3f)" %(epoch, result, iou))
            else:
                print("Batch %f test accuracy: %f" %(i, accuracy))
            total_err.append(result)

        #mean_loss = np.sum(total_loss) / num_examples
        #mean_accuracy = np.sum(total_accuracy)/num_examples
        #mean_accuracy = accSum / N
        #mean_error = np.sum(total_err) / num_examples
        mean_accuracy = np.mean(total_accuracy)
        print('average loss'+ str(np.mean(np.array(total_err)))+' Accuracy: '+str(mean_accuracy) + ' Error: ' + str(1 - mean_accuracy))
        return result, iou, mean_accuracy

if __name__ == '__main__':
    train_params_cam = {'batch_size': 2,
                        'n_epochs': 300,
                        'initial_learning_rate': 0.9,
                        'reduce_lr_epoch_1': 150,  # epochs * 0.5
                        'reduce_lr_epoch_2': 225,  # epochs * 0.75
                        'validation_set': False,
                        'validation_split': None,  # None or float
                        'shuffle': 'every_epoch',  # None, once_prior_train, every_epoch
                        'normalization': None,  # None, divide_256, divide_255, by_chanels
                        'augmentation': True
                        }

    dataset = cam.CamvidDataProvider(**train_params_cam)
    model = rnet(dataset)
    model.train_all_epochs(600)
