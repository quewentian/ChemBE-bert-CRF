# encoding=utf-8

import tensorflow as tf
from tensorflow.contrib import crf


class CRF(object):
    def __init__(self, embedded_chars, droupout_rate,
                 initializers,num_labels, seq_length, labels, lengths, is_training):

        self.droupout_rate = droupout_rate
        self.embedded_chars = embedded_chars
        self.initializers = initializers
        self.seq_length = seq_length
        self.num_labels = num_labels
        self.labels = labels
        self.lengths = lengths
        self.is_training = is_training

    def add_crf_layer(self):
        """
        crf网络
        :return:
        """
        if self.is_training:
            self.embedded_chars = tf.nn.dropout(self.embedded_chars, self.droupout_rate)
        # project
        logits = self.project_layer(self.embedded_chars)
        # crf
        loss, trans = self.crf_layer(logits)
        # CRF decode, pred_ids 是一条最大概率的标注路径
        pred_ids, _ = crf.crf_decode(potentials=logits, transition_params=trans, sequence_length=self.lengths)
        return ((loss, logits, trans, pred_ids))


    def project_layer(self, embedded_chars, name=None):
        """
        hidden layer between lstm layer and logits
        :param lstm_outputs: [batch_size, num_steps, emb_size]
        :return: [batch_size, num_steps, num_tags]
        """
        hidden_state = self.embedded_chars.get_shape()[-1]
        with tf.variable_scope("project" if not name else name):
            # project to score of tags
            with tf.variable_scope("logits"):
                W = tf.get_variable("W", shape=[hidden_state, self.num_labels],
                                    dtype=tf.float32, initializer=self.initializers.xavier_initializer())

                b = tf.get_variable("b", shape=[self.num_labels], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())

                embeddeding = tf.reshape(self.embedded_chars,[-1, hidden_state])
                pred = tf.nn.xw_plus_b(embeddeding, W, b)
            return tf.reshape(pred, [-1, self.seq_length, self.num_labels])


    def crf_layer(self, logits):
        """
        calculate crf loss
        :param project_logits: [1, num_steps, num_tags]
        :return: scalar loss
        """
        with tf.variable_scope("crf_loss"):
            trans = tf.get_variable(
                "transitions",
                shape=[self.num_labels, self.num_labels],
                initializer=self.initializers.xavier_initializer())
            log_likelihood, trans = tf.contrib.crf.crf_log_likelihood(
                inputs=logits,
                tag_indices=self.labels,
                transition_params=trans,
                sequence_lengths=self.lengths)
            return tf.reduce_mean(-log_likelihood), trans