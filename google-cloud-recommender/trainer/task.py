"""Job entry point for ML Engine."""

import argparse
import json
import os
import tensorflow as tf

import model
import util
import wals
import local_test


def main(args):
    # process input file
    input_file = util.ensure_local_file(args['train_files'][0])
    train_sparse, test_sparse, val_sparse, user_item_interactions, user_item_pref, val_users_idx, test_users_idx = model.create_test_and_train_sets(
      args, input_file, args['data_type'])

    # train model
    recommendation = model.train_model(args, train_sparse, val_sparse, user_item_interactions, val_users_idx)
    print('\n')

    # log results
    test_precision = model.top_k_precision(args['k_recommends'], recommendation, test_sparse, test_users_idx)
    print('The model has predicted {} recommendations for the users in the test set:'.format(args['k_recommends']))
    print('The average test set recommendation precision from the model = {:.2f}% match'.format(args['k_recommends'], test_precision * 100))
    print('\n')
    # To test the saved model with a JSON file:
    print('Local JSON test:')
    local_test.local_json_test()
#--------------------------------------

def parse_arguments():
  """Parse job arguments."""
  parser = argparse.ArgumentParser()
  # required input arguments
  parser.add_argument(
      '--train-files',
      help='GCS or local paths to training data',
      nargs='+',
      required=True
  )
  parser.add_argument(
      '--job-dir',
      help='GCS location to write checkpoints and export models',
      required=True
  )

  # hyper params for model
  parser.add_argument(
      '--latent_factors',
      type=int,
      help='Number of latent factors',
  )
  parser.add_argument(
      '--num_iters',
      type=int,
      help='Number of iterations for alternating least squares factorization',
  )
  parser.add_argument(
      '--regularization',
      type=float,
      help='L2 regularization factor',
  )

  # other args
  parser.add_argument(
      '--output-dir',
      help='GCS location to write model, overriding job-dir',
  )
  parser.add_argument(
      '--verbose-logging',
      default=False,
      action='store_true',
      help='Switch to turn on or off verbose logging and warnings'
  )
  parser.add_argument(
      '--hypertune',
      default=False,
      action='store_true',
      help='Switch to turn on or off hyperparam tuning'
  )
  parser.add_argument(
      '--data-type',
      type=str,
      default='ratings',
      help='Data type, one of ratings (e.g. MovieLens) or web_views (GA data)'
  )
  parser.add_argument(
      '--delimiter',
      type=str,
      default='\t',
      help='Delimiter for csv data files'
  )
  parser.add_argument(
      '--headers',
      default=False,
      action='store_true',
      help='Input file has a header row'
  )
  parser.add_argument(
      '--use-optimized',
      default=False,
      action='store_true',
      help='Use optimized hyperparameters'
  )

  args = parser.parse_args()
  arguments = args.__dict__

  # set job name as job directory name
  job_dir = args.job_dir
  job_dir = job_dir[:-1] if job_dir.endswith('/') else job_dir
  job_name = os.path.basename(job_dir)

  # set output directory for model
  if args.hypertune:
    # if tuning, join the trial number to the output path
    config = json.loads(os.environ.get('TF_CONFIG', '{}'))
    trial = config.get('task', {}).get('trial', '')
    output_dir = os.path.join(job_dir, trial)
  elif args.output_dir:
    output_dir = args.output_dir
  else:
    output_dir = job_dir

  if args.verbose_logging:
    tf.logging.set_verbosity(tf.logging.INFO)

  # Find out if there's a task value on the environment variable.
  # If there is none or it is empty define a default one.
  env = json.loads(os.environ.get('TF_CONFIG', '{}'))
  task_data = env.get('task') or {'type': 'master', 'index': 0}

  # update default params with any args provided to task
  params = model.DEFAULT_PARAMS
  params.update({k: arg for k, arg in arguments.iteritems() if arg is not None})
  if args.use_optimized:
    if args.data_type == 'web_views':
      params.update(model.OPTIMIZED_PARAMS_WEB)
    else:
      params.update(model.OPTIMIZED_PARAMS)
  params.update(task_data)
  params.update({'output_dir': output_dir})
  params.update({'job_name': job_name})

  # For web_view data, default to using the exponential weight formula
  # with feature weight exp.
  # For movie lens data, default to the linear weight formula.
  if args.data_type == 'web_views':
    params.update({'wt_type': wals.LOG_RATINGS})

  return params
#--------------------------------------

if __name__ == '__main__':
  job_args = parse_arguments()
  main(job_args)