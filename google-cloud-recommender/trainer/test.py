#This function use a JSON file to recommend items for the user.

from model import generate_recommendations
import json
import numpy as np
import os
import task
from StringIO import StringIO
from tensorflow.python.lib.io import file_io
import tensorflow as tf



def main(args):
    model_dir = os.path.join(args['output_dir'])
    rec_path = os.path.join(args['output_dir'])
    pref_path = os.path.join(args['output_dir'])
    print("The args['output_dir'] = {}".format(args['output_dir']))
    if model_dir.startswith('gs://'):
        print("gc path = {}".format(os.path.join(model_dir, 'model', 'recommendation_matrix.npy')))
        rec_path = os.path.join(model_dir, 'model', 'recommendation_matrix.npy')
        pref_path = os.path.join(model_dir, 'model', 'Preference_matrix.npy')

    else:
        # model_dir here is the path for the latest recommendation matrix which was previously trained
        model_dir = (os.popen("cd './jobs' && ls -t | head -1").readlines())[0].strip( '\n' )
        # Loading the matrix from the cmdOutput1 path
        rec_path = os.path.join('jobs', model_dir, 'model', 'recommendation_matrix.npy')
        pref_path = os.path.join('jobs', model_dir, 'model', 'Preference_matrix.npy')

    print("gc path 1 = {}".format(rec_path))

    sess = tf.Session()
    with sess.as_default():
        f_r = StringIO(file_io.read_file_to_string(rec_path))
        print("gc f path = {}".format(f_r))
        recommendation_numpy = tf.constant(np.load(f_r), name='recommendation_numpy')
        print("gc my_variable = {}".format(recommendation_numpy.eval()))

        f_p = StringIO(file_io.read_file_to_string(pref_path))
        Preference_numpy = tf.constant(np.load(f_p), name='Preference_numpy')


        recommendation_matrix = recommendation_numpy.eval()
        preference_matrix = Preference_numpy.eval()

        print("preference_matrix= {}".format(preference_matrix))


        with file_io.FileIO('gs://azg_bucket/data/test.json', mode='r') as f:
            data = json.load(f)
            users = data["user_id"]
            for user_id in users:
                recommendations = generate_recommendations(100, user_id, recommendation_matrix, preference_matrix)
                print("Recommendations for user {} = {}".format(user_id, recommendations))

        #sess.close()

if __name__ == '__main__':
    job_args = task.parse_arguments()
    main(job_args)
