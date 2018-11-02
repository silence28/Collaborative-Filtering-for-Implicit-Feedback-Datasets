"""This script is used to load a tensorflow model
   and test it with a JSON file locally without the cloud"""

import os
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants

def local_json_test():
    graph = tf.Graph()
    with graph.as_default():
        model_dir = (os.popen("cd './jobs' && ls -t | head -1 ").readlines())[0].strip('\n')
        model_dir = os.path.join('jobs', model_dir, 'model')
        sess=tf.Session()
        tf.saved_model.loader.load(
                     sess,
                     [tag_constants.SERVING],
                     model_dir
                 )
        User_id_for_recommendation = graph.get_tensor_by_name('User_id_for_recommendation:0')
        k_recommendations = graph.get_tensor_by_name('k_recommendations:0')
        recommendations_for_user_id = graph.get_tensor_by_name('recommendations_for_user_id:0')

        sess.run(recommendations_for_user_id, feed_dict={
                     User_id_for_recommendation: 4, #User id
                     k_recommendations: 10 # Number of recommendations for the user
        })

        with tf.Session() as sess:
            result = recommendations_for_user_id
            print("{} Items recommended for user_id: {}, Items = {} ".format(10, 4, sess.run(result)))

