from sklearn.metrics import f1_score

def compute_f1(learner, X, y_true, average=None):
    y_pred = learner.predict(X)
    return f1_score(y_true, y_pred, average=average)