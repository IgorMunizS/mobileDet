# Project: squeezeDetOnKeras
# Filename: squeezeDet
# Author: Christopher Ehmann
# Date: 28.11.17
# Organisation: searchInk
# Email: christopher@searchink.com


from keras.models import Model
from keras.layers import Input, MaxPool2D, Conv2D, Dropout, concatenate, Reshape, Lambda, AveragePooling2D, Activation
from keras import backend as K
from keras.initializers import TruncatedNormal, random_normal, constant
from keras.regularizers import l2
import main.utils.utils as utils
import numpy as np
import tensorflow as tf
from keras import layers


# class that wraps config and model
class CMUDet():
    # initialize model from config file
    def __init__(self, config):
        """Init of SqueezeDet Class

        Arguments:
            config {[type]} -- dict containing hyperparameters for network building
        """

        # hyperparameter config file
        self.config = config
        # create Keras model
        self.model = self._create_model()

    # creates keras model
    def _create_model(self, alpha=0.75, depth_multiplier=1):
        """
        #builds the Keras model from config
        #return: mobileDet in Keras

        """
        weight_decay = 0.001
        input_layer = Input(shape=(self.config.IMAGE_HEIGHT, self.config.IMAGE_WIDTH, self.config.N_CHANNELS),name="input")

        # Block 1
        x = self.conv(input_layer, 64, 3, "conv1_1", (weight_decay, 0))
        x = self.relu(x)
        x = self.conv(x, 64, 3, "conv1_2", (weight_decay, 0))
        x = self.relu(x)
        x = self.pooling(x, 2, 2, "pool1_1")

        # Block 2
        x = self.conv(x, 128, 3, "conv2_1", (weight_decay, 0))
        x = self.relu(x)
        x = self.conv(x, 128, 3, "conv2_2", (weight_decay, 0))
        x = self.relu(x)
        x = self.pooling(x, 2, 2, "pool2_1")

        # Block 3
        x = self.conv(x, 256, 3, "conv3_1", (weight_decay, 0))
        x = self.relu(x)
        x = self.conv(x, 256, 3, "conv3_2", (weight_decay, 0))
        x = self.relu(x)
        x = self.conv(x, 256, 3, "conv3_3", (weight_decay, 0))
        x = self.relu(x)
        x = self.conv(x, 256, 3, "conv3_4", (weight_decay, 0))
        x = self.relu(x)
        x = self.pooling(x, 2, 2, "pool3_1")

        # Block 4
        x = self.conv(x, 512, 3, "conv4_1", (weight_decay, 0))
        x = self.relu(x)
        x = self.conv(x, 512, 3, "conv4_2", (weight_decay, 0))
        x = self.relu(x)

        # Additional non vgg layers
        x = self.conv(x, 256, 3, "conv4_3_CPM", (weight_decay, 0))
        x = self.relu(x)
        x = self.conv(x, 128, 3, "conv4_4_CPM", (weight_decay, 0))
        x = self.relu(x)

        dropout11 = Dropout(rate=self.config.KEEP_PROB, name='drop11')(x)

        # compute the number of output nodes from number of anchors, classes, confidence score and bounding box corners
        num_output = self.config.ANCHOR_PER_GRID * (self.config.CLASSES + 1 + 4)

        preds = Conv2D(
            name='conv12', filters=num_output, kernel_size=(3, 3), strides=(1, 1), activation=None, padding="SAME",
            use_bias=True, kernel_initializer=TruncatedNormal(stddev=0.001),
            kernel_regularizer=l2(self.config.WEIGHT_DECAY))(dropout11)

        # reshape
        pred_reshaped = Reshape((self.config.ANCHORS, -1))(preds)

        # pad for loss function so y_pred and y_true have the same dimensions, wrap in lambda layer
        pred_padded = Lambda(self._pad)(pred_reshaped)

        model = Model(inputs=input_layer, outputs=pred_padded)
        print(model.summary())
        return model

    def relu(self,x):
        return Activation('relu')(x)

    def conv(self,x, nf, ks, name, weight_decay):
        kernel_reg = l2(weight_decay[0]) if weight_decay else None
        bias_reg = l2(weight_decay[1]) if weight_decay else None

        x = Conv2D(nf, (ks, ks), padding='same', name=name,
                   kernel_regularizer=kernel_reg,
                   bias_regularizer=bias_reg,
                   kernel_initializer=random_normal(stddev=0.01),
                   bias_initializer=constant(0.0))(x)
        return x

    def pooling(self, x, ks, st, name):
        x = MaxPool2D((ks, ks), strides=(st, st), name=name)(x)
        return x


    # wrapper for padding, written in tensorflow. If you want to change to theano you need to rewrite this!
    def _pad(self, input):
        """
        pads the network output so y_pred and y_true have the same dimensions
        :param input: previous layer
        :return: layer, last dimensions padded for 4
        """

        # pad = K.placeholder( (None,self.config.ANCHORS, 4))

        # pad = np.zeros ((self.config.BATCH_SIZE,self.config.ANCHORS, 4))
        # return K.concatenate( [input, pad], axis=-1)

        padding = np.zeros((3, 2))
        padding[2, 1] = 4
        return tf.pad(input, padding, "CONSTANT")

    # loss function to optimize
    def loss(self, y_true, y_pred):
        """
        squeezeDet loss function for object detection and classification
        :param y_true: ground truth with shape [batchsize, #anchors, classes+8+labels]
        :param y_pred:
        :return: a tensor of the total loss
        """

        # handle for config
        mc = self.config

        # slice y_true
        input_mask = y_true[:, :, 0]
        input_mask = K.expand_dims(input_mask, axis=-1)
        box_input = y_true[:, :, 1:5]
        box_delta_input = y_true[:, :, 5:9]
        labels = y_true[:, :, 9:]

        # number of objects. Used to normalize bbox and classification loss
        num_objects = K.sum(input_mask)

        # before computing the losses we need to slice the network outputs
        pred_class_probs, pred_conf, pred_box_delta = utils.slice_predictions(y_pred, mc)

        # compute boxes
        det_boxes = utils.boxes_from_deltas(pred_box_delta, mc)

        # again unstack is not avaible in pure keras backend
        unstacked_boxes_pred = []
        unstacked_boxes_input = []

        for i in range(4):
            unstacked_boxes_pred.append(det_boxes[:, :, i])
            unstacked_boxes_input.append(box_input[:, :, i])

        # compute the ious
        ious = utils.tensor_iou(utils.bbox_transform(unstacked_boxes_pred),
                                utils.bbox_transform(unstacked_boxes_input),
                                input_mask,
                                mc
                                )

        # compute class loss,add a small value into log to prevent blowing up
        class_loss = K.sum(labels * (-K.log(pred_class_probs + mc.EPSILON))
                           + (1 - labels) * (-K.log(1 - pred_class_probs + mc.EPSILON))
                           * input_mask * mc.LOSS_COEF_CLASS) / num_objects

        # bounding box loss
        bbox_loss = (K.sum(mc.LOSS_COEF_BBOX * K.square(input_mask * (pred_box_delta - box_delta_input))) / num_objects)

        # reshape input for correct broadcasting
        input_mask = K.reshape(input_mask, [mc.BATCH_SIZE, mc.ANCHORS])

        # confidence score loss
        conf_loss = K.mean(
            K.sum(
                K.square((ious - pred_conf))
                * (input_mask * mc.LOSS_COEF_CONF_POS / num_objects
                   + (1 - input_mask) * mc.LOSS_COEF_CONF_NEG / (mc.ANCHORS - num_objects)),
                axis=[1]
            ),
        )

        # add above losses
        total_loss = class_loss + conf_loss + bbox_loss

        return total_loss

    # the sublosses, to be used as metrics during training

    def bbox_loss(self, y_true, y_pred):
        """
        squeezeDet loss function for object detection and classification
        :param y_true: ground truth with shape [batchsize, #anchors, classes+8+labels]
        :param y_pred:
        :return: a tensor of the bbox loss
        """

        # handle for config
        mc = self.config

        # calculate non padded entries
        n_outputs = mc.CLASSES + 1 + 4

        # slice and reshape network output
        y_pred = y_pred[:, :, 0:n_outputs]
        y_pred = K.reshape(y_pred, (mc.BATCH_SIZE, mc.N_ANCHORS_HEIGHT, mc.N_ANCHORS_WIDTH, -1))

        # slice y_true
        input_mask = y_true[:, :, 0]
        input_mask = K.expand_dims(input_mask, axis=-1)
        box_delta_input = y_true[:, :, 5:9]

        # number of objects. Used to normalize bbox and classification loss
        num_objects = K.sum(input_mask)

        # before computing the losses we need to slice the network outputs

        # number of class probabilities, n classes for each anchor
        num_class_probs = mc.ANCHOR_PER_GRID * mc.CLASSES

        # number of confidence scores, one for each anchor + class probs
        num_confidence_scores = mc.ANCHOR_PER_GRID + num_class_probs

        # slice the confidence scores and put them trough a sigmoid for probabilities
        pred_conf = K.sigmoid(
            K.reshape(
                y_pred[:, :, :, num_class_probs:num_confidence_scores],
                [mc.BATCH_SIZE, mc.ANCHORS]
            )
        )

        # slice remaining bounding box_deltas
        pred_box_delta = K.reshape(
            y_pred[:, :, :, num_confidence_scores:],
            [mc.BATCH_SIZE, mc.ANCHORS, 4]
        )

        # cross-entropy: q * -log(p) + (1-q) * -log(1-p)
        # add a small value into log to prevent blowing up

        # bounding box loss
        bbox_loss = (K.sum(mc.LOSS_COEF_BBOX * K.square(input_mask * (pred_box_delta - box_delta_input))) / num_objects)

        return bbox_loss

    def conf_loss(self, y_true, y_pred):
        """
        squeezeDet loss function for object detection and classification
        :param y_true: ground truth with shape [batchsize, #anchors, classes+8+labels]
        :param y_pred:
        :return: a tensor of the conf loss
        """

        # handle for config
        mc = self.config

        # calculate non padded entries
        n_outputs = mc.CLASSES + 1 + 4

        # slice and reshape network output
        y_pred = y_pred[:, :, 0:n_outputs]
        y_pred = K.reshape(y_pred, (mc.BATCH_SIZE, mc.N_ANCHORS_HEIGHT, mc.N_ANCHORS_WIDTH, -1))

        # slice y_true
        input_mask = y_true[:, :, 0]
        input_mask = K.expand_dims(input_mask, axis=-1)
        box_input = y_true[:, :, 1:5]

        # number of objects. Used to normalize bbox and classification loss
        num_objects = K.sum(input_mask)

        # before computing the losses we need to slice the network outputs

        # number of class probabilities, n classes for each anchor
        num_class_probs = mc.ANCHOR_PER_GRID * mc.CLASSES

        # number of confidence scores, one for each anchor + class probs
        num_confidence_scores = mc.ANCHOR_PER_GRID + num_class_probs

        # slice the confidence scores and put them trough a sigmoid for probabilities
        pred_conf = K.sigmoid(
            K.reshape(
                y_pred[:, :, :, num_class_probs:num_confidence_scores],
                [mc.BATCH_SIZE, mc.ANCHORS]
            )
        )

        # slice remaining bounding box_deltas
        pred_box_delta = K.reshape(
            y_pred[:, :, :, num_confidence_scores:],
            [mc.BATCH_SIZE, mc.ANCHORS, 4]
        )

        # compute boxes
        det_boxes = utils.boxes_from_deltas(pred_box_delta, mc)

        # again unstack is not avaible in pure keras backend
        unstacked_boxes_pred = []
        unstacked_boxes_input = []

        for i in range(4):
            unstacked_boxes_pred.append(det_boxes[:, :, i])
            unstacked_boxes_input.append(box_input[:, :, i])

        # compute the ious
        ious = utils.tensor_iou(utils.bbox_transform(unstacked_boxes_pred),
                                utils.bbox_transform(unstacked_boxes_input),
                                input_mask,
                                mc
                                )

        # reshape input for correct broadcasting
        input_mask = K.reshape(input_mask, [mc.BATCH_SIZE, mc.ANCHORS])

        # confidence score loss
        conf_loss = K.mean(
            K.sum(
                K.square((ious - pred_conf))
                * (input_mask * mc.LOSS_COEF_CONF_POS / num_objects
                   + (1 - input_mask) * mc.LOSS_COEF_CONF_NEG / (mc.ANCHORS - num_objects)),
                axis=[1]
            ),
        )

        return conf_loss

    def class_loss(self, y_true, y_pred):
        """
        squeezeDet loss function for object detection and classification
        :param y_true: ground truth with shape [batchsize, #anchors, classes+8+labels]
        :param y_pred:
        :return: a tensor of the class loss
        """

        # handle for config
        mc = self.config

        # calculate non padded entries
        n_outputs = mc.CLASSES + 1 + 4

        # slice and reshape network output
        y_pred = y_pred[:, :, 0:n_outputs]
        y_pred = K.reshape(y_pred, (mc.BATCH_SIZE, mc.N_ANCHORS_HEIGHT, mc.N_ANCHORS_WIDTH, -1))

        # slice y_true
        input_mask = y_true[:, :, 0]
        input_mask = K.expand_dims(input_mask, axis=-1)
        labels = y_true[:, :, 9:]

        # number of objects. Used to normalize bbox and classification loss
        num_objects = K.sum(input_mask)

        # before computing the losses we need to slice the network outputs

        # number of class probabilities, n classes for each anchor
        num_class_probs = mc.ANCHOR_PER_GRID * mc.CLASSES

        # slice pred tensor to extract class pred scores and then normalize them
        pred_class_probs = K.reshape(
            K.softmax(
                K.reshape(
                    y_pred[:, :, :, :num_class_probs],
                    [-1, mc.CLASSES]
                )
            ),
            [mc.BATCH_SIZE, mc.ANCHORS, mc.CLASSES],
        )

        # cross-entropy: q * -log(p) + (1-q) * -log(1-p)
        # add a small value into log to prevent blowing up

        # compute class loss
        class_loss = K.sum((labels * (-K.log(pred_class_probs + mc.EPSILON))
                            + (1 - labels) * (-K.log(1 - pred_class_probs + mc.EPSILON)))
                           * input_mask * mc.LOSS_COEF_CLASS) / num_objects

        return class_loss

    # loss function again, used for metrics to show loss without regularization cost, just of copy of the original loss
    def loss_without_regularization(self, y_true, y_pred):
        """
        squeezeDet loss function for object detection and classification
        :param y_true: ground truth with shape [batchsize, #anchors, classes+8+labels]
        :param y_pred:
        :return: a tensor of the total loss
        """

        # handle for config
        mc = self.config

        # slice y_true
        input_mask = y_true[:, :, 0]
        input_mask = K.expand_dims(input_mask, axis=-1)
        box_input = y_true[:, :, 1:5]
        box_delta_input = y_true[:, :, 5:9]
        labels = y_true[:, :, 9:]

        # number of objects. Used to normalize bbox and classification loss
        num_objects = K.sum(input_mask)

        # before computing the losses we need to slice the network outputs

        pred_class_probs, pred_conf, pred_box_delta = utils.slice_predictions(y_pred, mc)

        # compute boxes
        det_boxes = utils.boxes_from_deltas(pred_box_delta, mc)

        # again unstack is not avaible in pure keras backend
        unstacked_boxes_pred = []
        unstacked_boxes_input = []

        for i in range(4):
            unstacked_boxes_pred.append(det_boxes[:, :, i])
            unstacked_boxes_input.append(box_input[:, :, i])

        # compute the ious
        ious = utils.tensor_iou(utils.bbox_transform(unstacked_boxes_pred),
                                utils.bbox_transform(unstacked_boxes_input),
                                input_mask,
                                mc)

        # cross-entropy: q * -log(p) + (1-q) * -log(1-p)
        # add a small value into log to prevent blowing up

        # compute class loss
        class_loss = K.sum(labels * (-K.log(pred_class_probs + mc.EPSILON))
                           + (1 - labels) * (-K.log(1 - pred_class_probs + mc.EPSILON))
                           * input_mask * mc.LOSS_COEF_CLASS) / num_objects

        # bounding box loss
        bbox_loss = (K.sum(mc.LOSS_COEF_BBOX * K.square(input_mask * (pred_box_delta - box_delta_input))) / num_objects)

        # reshape input for correct broadcasting
        input_mask = K.reshape(input_mask, [mc.BATCH_SIZE, mc.ANCHORS])

        # confidence score loss
        conf_loss = K.mean(
            K.sum(
                K.square((ious - pred_conf))
                * (input_mask * mc.LOSS_COEF_CONF_POS / num_objects
                   + (1 - input_mask) * mc.LOSS_COEF_CONF_NEG / (mc.ANCHORS - num_objects)),
                axis=[1]
            ),
        )

        # add above losses
        total_loss = class_loss + conf_loss + bbox_loss

        return total_loss