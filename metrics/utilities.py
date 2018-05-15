
def log_loss_fn(y_true, y_pred, 
                labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]):
  return log_loss(inverse_onehot(y_true), y_pred.tolist(), eps=1e-15, labels=labels)


def neg_log_loss(y_true, y_pred,
                 labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]):
  return log_loss(y_true, y_pred, eps=1e-15, labels=labels)


def f1_score_objective(y_true, y_pred,
                       labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                       average='weighted',
                       sample_weight=None):
  return f1_score(y_true, y_pred, labels=labels,
                  average=average, sample_weight=sample_weight)


def geometric_mean(y_true, y_pred, average='weighted', #'multiclass'
                   labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 
                   sample_weight=None, correction=0.0):
  return geometric_mean_score(y_true, y_pred, labels=labels, average=average, 
                              sample_weight=sample_weight, correction=correction)

def categorical_crossentropy_loss(y_true, y_pred):
  # converg predicted and true y targets ndarray into tensorflow tensor objects
  y_true = tf.convert_to_tensor(inverse_onehot(y_true), dtype=tf.float64)
  y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float64)

  # calculate categorical crossy entropy (log-loss)
  score = K.categorical_crossentropy(y_true, y_pred).eval(session=K.get_session())
  #K.clear_session()

  # check score value is not NaN
  if math.isnan(score) or not isinstance(score, float):
      score = 1.0e7
  #print('categorical_crossentropy', score)
    
  return score
