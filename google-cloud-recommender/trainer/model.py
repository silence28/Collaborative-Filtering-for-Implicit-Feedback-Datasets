"""This file prepares the data, train and tests it"""

import datetime
import numpy as np
import pandas as pd
import os
import tensorflow as tf


INPUT_COLUMNS = [
    'User_id_for_recommendation',
    'k_recommendations'
]
#--------------------------------------------------------------------

# default hyperparameters
DEFAULT_PARAMS = {
    'k_recommends': 10,
    'regularization': 0.01,
    'latent_factors': 10,
    'num_iters': 80,
    'learning_rate': 0.05,
    'delimiter': '\t'
}
#--------------------------------------------------------------------

def create_test_and_train_sets(args, input_file, data_type='ratings'):
  """Create test and train sets, for different input data types.

  Args:
    args: input args for job
    input_file: path to csv data file
    data_type:  Different type of data can be chosen here.
  """
  if data_type == 'ratings':
    return _ratings_train_and_test(args,args['headers'], args['delimiter'],
                                   input_file)
  else:
    raise ValueError('data_type arg value %s not supported.' % data_type)

#--------------------------------------------------------------------
def data_process(interaction, preference, train, test, user_idx, k):
    """The function is used to mask the preferences data from training matrix.
    Args:
        interaction: User Item interactions matrix where it contains interaction between the item and the user.
        preference:  User Item preference matrix which tells us which item has been bought by the user.
        train: copy of User Item preference matrix
        test: test set matrix which is the copy of User Item preference matrix and will be masked in this function
        user_idx: list of user for which their data will be masked
        k: number of recommendation to predict
    return: returns train set, test set and interaction matrix
    """
    for user in user_idx:
        purchases = np.where(preference[user, :] == 1)[0]
        mask = np.random.choice(purchases, size=k, replace=False)
        interaction[user, mask] = 0
        train[user, mask] = 0
        test[user, mask] = preference[user, mask]
    return train, test, interaction
#--------------------------------------------------------------------

def _ratings_train_and_test(args,use_headers, delimiter, input_file):
    """Load data set.
    Args:
        use_headers: (boolean) true = headers, false = no headers
        delimiter: (string) delimiter to use for csv
        input_file: path to csv data file

     Returns:
        train_sparse: training set sparse matrix
        test_sparse: test set sparse matrix
        val_sparse: validation set sparse matrix
        user_item_interactions:
        User_Item_pref:
        val_users_idx: validation set users
        test_users_idx: validation set users
    """
    headers = ['user_id', 'item_id', 'click', 'timestamp']
    header_row = 0 if use_headers else None
    df = pd.read_csv(input_file,
                           sep=delimiter,
                           names=headers,
                           header=header_row,
                           dtype={
                               'user_id': np.int32,
                               'item_id': np.int32,
                               'click': np.float32,
                               'timestamp': np.int32,
                           })
    df = df.drop('timestamp', axis=1)  # Removing Timestamp

    # Sort the df by user_id, item_id, and click
    df_Sorted = df.sort_values(['user_id', 'item_id', 'click'])

    # Drop the duplicated records (If someone watched the same item twice):
    clean_df = df_Sorted.drop_duplicates(['user_id', 'item_id'], keep='last')

    n_user = len(clean_df.user_id.unique())
    n_item = len(clean_df.item_id.unique())

    # User Item preference matrix:
    User_Item_pref = clean_df.copy()
    User_Item_pref['click'][User_Item_pref['click'] > 0] = 1
    User_Item_pref = User_Item_pref.pivot(index='user_id', columns='item_id', values='click')
    User_Item_pref.fillna(0, inplace=True)
    User_Item_pref = User_Item_pref.values

    # User Item interaction matrix:
    User_Item_interactions = clean_df.pivot(index='user_id', columns='item_id', values='click')
    User_Item_interactions.fillna(0, inplace=True)
    User_Item_interactions = User_Item_interactions.values

    k = args['k_recommends']  # Number of top k items we want to recommend for the user

    # View_counts counts the number of item viewed by each user:
    View_counts = np.apply_along_axis(np.bincount, 1, User_Item_pref.astype(int))

    # buyers_idx finds the users who purchased 2*k items:
    buyers_idx = np.where(View_counts[:, 1] >= k * 2)[0]

    # Let's save 10% of the data for validation and 10% for testing:
    test_frac = 0.2
    test_users_idx = np.random.choice(buyers_idx,
                                    size=int(np.ceil(len(buyers_idx) * test_frac)),
                                    replace=False)

    val_users_idx = test_users_idx[:int(len(test_users_idx) / 2)]
    test_users_idx = test_users_idx[int(len(test_users_idx) / 2):]

    zero_matrix = np.zeros(shape=(n_user, n_item))
    train_matrix = User_Item_pref.copy()
    test_matrix = zero_matrix.copy()
    val_matrix = zero_matrix.copy()

    # Mask the train matrix and create the validation and test matrices
    train_sparse, val_sparse, user_item_interactions = data_process(User_Item_interactions,
                                                                  User_Item_pref, train_matrix,
                                                                  val_matrix, val_users_idx, k)
    train_sparse, test_sparse, user_item_interactions = data_process(User_Item_interactions,
                                                                   User_Item_pref, train_matrix,
                                                                   test_matrix, test_users_idx, k)
    return train_sparse, test_sparse, val_sparse, user_item_interactions, User_Item_pref, val_users_idx, test_users_idx

#--------------------------------------------------------------------
def top_k_precision(k, predicted, mat, user_idx):
  """This is a function that helps to calculate the top k item average predicted precisions"""
  precisions = []

  for user in user_idx:
    rec = np.argsort(-predicted[user, :]) # The argsort sorts the recommendation
                                            #for each user from high to low
    top_k = rec[:k] # Getting the top k items
    labels = mat[user, :].nonzero()[0]
    precision = len(set(top_k) & set(labels)) / float(k)  # Calculate the precisions from actual labels
    precisions.append(precision)
  return np.mean(precisions)
#--------------------------------------------------------------------
def train_model(args, train_sparse, val_sparse, user_item_interactions, val_users_idx):
  """This function will train the model.

  Args:
    args: training args containing hyperparams
    tr_sparse: sparse training matrix
    val_sparse: validation set sparse matrix
    user_item_interactions
    val_users_idx: validation set users

  Returns:
    recommendation: This produce a matrix where the rows are the user and
                    the columns are the expected recommendation for each movie
  """
  tf.logging.info('Train Start: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))
  tf.reset_default_graph()  # Create a new graphs


  #
  n_features = args['latent_factors'] # Number of latent features to be extracted
  iterations = args['num_iters']
  lambda_c = args['regularization'] # Regularization constant
  lr = args['learning_rate'] # learning rate
  k = args['k_recommends'] # Number of recommendations


  n_user = train_sparse.shape[0] # Getting the number of users
  n_item = train_sparse.shape[1] # Getting the number of items
  pref = tf.placeholder(tf.float32, (n_user, n_item), name = 'pref')  # Here's the preference matrix
  interactions = tf.placeholder(tf.float32, (n_user, n_item), name = 'interactions')  # Here viewed or not viewed matrix


  # The X matrix represents the user latent preferences with a shape of user x latent features
  X = tf.Variable(tf.truncated_normal([n_user, n_features], mean=0, stddev=0.05))

  # The Y matrix represents the item latent features with a shape of item x latent features
  Y = tf.Variable(tf.truncated_normal([n_item, n_features], mean=0, stddev=0.05))

  # Here's the initilization of the confidence parameter
  conf_alpha = tf.Variable(tf.random_uniform([1], 0, 1))

  # Initialize a user bias vector n_User, n_Product
  user_bias = tf.Variable(tf.truncated_normal([n_user, 1], stddev=0.2))

  # Concatenate the vector to the user matrix
  # Due to how matrix algebra works, we also need to add a column of ones to make sure
  # the resulting calculation will take into account the item biases.
  X_plus_bias = tf.concat([X,
                           # tf.convert_to_tensor(user_bias, dtype = tf.float32),
                           user_bias,
                           tf.ones((n_user, 1), dtype=tf.float32)], axis=1)

  # Initialize the item bias vector:
  item_bias = tf.Variable(tf.truncated_normal([n_item, 1], stddev=0.2))

  # Cocatenate the vector to the item matrix
  # Also, adds a column one for the same reason stated above.
  Y_plus_bias = tf.concat([Y,
                           tf.ones((n_item, 1), dtype=tf.float32),
                           item_bias],
                          axis=1)

  # Here, we finally multiply the matrices together to estimate the predicted preferences
  pred_pref = tf.matmul(X_plus_bias, Y_plus_bias, transpose_b=True)

  # Construct the confidence matrix with the clicks and alpha paramter
  conf = 1 + conf_alpha * interactions

  cost = tf.reduce_sum(tf.multiply(conf, tf.square(tf.subtract(pref, pred_pref))))
  l2_sqr = tf.nn.l2_loss(X) + tf.nn.l2_loss(Y) + tf.nn.l2_loss(user_bias) + tf.nn.l2_loss(item_bias)

  loss = cost + lambda_c * l2_sqr

  optimize = tf.train.AdagradOptimizer(learning_rate=lr).minimize(loss)

  # Defining a tensor for user and the numbers of recommendations with names. It is useful when exporting the model.
  User_id_for_recommendation = tf.Variable(2, name='User_id_for_recommendation') # Here User_id_for_recommendation = 2,
                                                                 # is for illustration purpose so the script can recommend for user_id = 2
  k_recommendations = tf.Variable(k, name='k_recommendations')

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(iterations):
      sess.run(optimize, feed_dict={pref: train_sparse,
                                    interactions: user_item_interactions})

      if i % 30 == 0: # For printing training Loss through iterations
        mod_loss = sess.run(loss, feed_dict={pref: train_sparse,
                                             interactions: user_item_interactions})
        mod_pred = pred_pref.eval()
        train_precision = top_k_precision(k, mod_pred, train_sparse, val_users_idx)
        val_precision = top_k_precision(k, mod_pred, val_sparse, val_users_idx)
        print('Iterations {0}...'.format(i),
              'Training Loss {:.2f}...'.format(mod_loss),
              'Train Precision {:.3f}...'.format(train_precision),
              'Val Precision {:.3f}'.format(val_precision)
         )
    print('\n')

    recommendation = pred_pref.eval()

    #--------
    # This part recommends items to a single user:
    rec_items = tf.contrib.framework.argsort(recommendation, -1, 'DESCENDING')
    purchase_history = tf.transpose(tf.where(train_sparse[User_id_for_recommendation.eval(), :] != 0))
    recommendations = rec_items[User_id_for_recommendation.eval(), :]
    a = recommendations.eval().astype(np.int64)
    b = purchase_history.eval().astype(np.int64)[0]
    a0 = tf.expand_dims(a, 1)
    b0 = tf.expand_dims(b, 0)
    tensor = a
    mask = ~tf.reduce_any(tf.equal(a0, b0), 1)
    rec_for_user_id = tf.boolean_mask(tensor, mask)[:k_recommendations.eval()]

    # Giving a name to the recommendations_for_user_id tensor so it can be restored from the model:
    recommendations_for_user_id = tf.identity(rec_for_user_id, name="recommendations_for_user_id")

    model_dir = os.path.join(args['output_dir'], 'model')

    inputs = {
            "User_id_for_recommendation": User_id_for_recommendation,
            "k_recommendations": k_recommendations}

    outputs = {"recommendations_for_user_id": recommendations_for_user_id}
    tf.saved_model.simple_save(
            sess, model_dir, inputs, outputs
    )

    tf.logging.info('Train Finish: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))
    sess.close()
  return recommendation
#--------------------------------------------------------------------

def generate_recommendations(k, user_id, recommendation, user_item_pref):
  """Generating k recommendations for a user.

  Args:
    k: number of recommendations requested

  Returns:
    list of k item indexes with the predicted highest rating, excluding
    those that the user has already rated
  """

  rec_items = np.argsort(-recommendation)

  purchase_history = np.where(user_item_pref[user_id, :] != 0)[0]
  recommendations = rec_items[user_id, :]

  new_recommendations = recommendations[~np.in1d(recommendations, purchase_history)][:k]

  return new_recommendations

#--------------------------------------------------------------------
# Functions below should not be touched!
#--------------------------------------------------------------------

# [START serving-function]
def json_serving_input_fn():
  """Build the serving inputs."""
  inputs = {}
  for feat in INPUT_COLUMNS:
    inputs[feat.name] = tf.placeholder(shape=[None], dtype=feat.dtype)

  return tf.estimator.export.ServingInputReceiver(inputs, inputs)

# [END serving-function]
SERVING_FUNCTIONS = {
    'JSON': json_serving_input_fn
}
