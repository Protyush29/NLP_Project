from sklearn.linear_model import SGDClassifier
from Language_models.doc2vec import Doc2Vec
from shared.data_visualization import  Plot

class Classifier:
    def __init__(self):
        self.classifier = SGDClassifier(loss='log', penalty='l1')

    def configure_classifier(self,model,train_vecs,test_vecs,y_train,y_test):
        self.classifier.fit(train_vecs, y_train)
        self.classifier.score(test_vecs, y_test)
        pred_probas = model.predict_proba(test_vecs)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, pred_probas)
        roc_auc = auc(fpr, tpr)

        plot = Plot()
        plot.plot_ROC(fpr,tpr,roc_auc)
