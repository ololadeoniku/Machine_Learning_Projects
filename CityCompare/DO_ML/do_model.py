#!/usr/bin/env python3

import DO_ML
from DO_ML.do_pre_process import *
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
# from matplotlib.ticker import StrMethodFormatter
import matplotlib.pyplot as plt


class Model:

    def __init__(self, xtrain_array, xtest_array, ytrain_array, ytest_array, model, target_mapper={}):
        self.xtrain = xtrain_array
        self.xtest = xtest_array
        self.ytrain = ytrain_array
        self.ytest = ytest_array
        self.model = model
        self.target_mapper = target_mapper
        self.ytest_mapped = pd.Series(self.ytest).map(self.target_mapper)

    def fit_model(self):
        self.model.fit(self.xtrain, self.ytrain)

    def make_prediction(self):
        return self.model.predict(self.xtest)


class RegressionModel(Model):

    def __init__(self, xtrain_array, xtest_array, ytrain_array, ytest_array, model, target_mapper={}):
        super().__init__(xtrain_array, xtest_array, ytrain_array, ytest_array, model, target_mapper)

    def get_coefficient(self):
        return self.model.coef_

    def get_intercept(self):
        return self.model.intercept_

    def get_score(self):
        return self.model.score(self.xtrain, self.ytrain)

class ClassificationModel(Model):

    def __init__(self, xtrain_array, xtest_array, ytrain_array, ytest_array, model, target_mapper={}):
        super().__init__(xtrain_array, xtest_array, ytrain_array, ytest_array, model, target_mapper)

    def draw_confusion_matrix(self, scaled=None, size=5):
        # fig, ax = plt.subplots(figsize=(size, size))
        plot_confusion_matrix(self.model, self.xtest, self.ytest, labels=None, sample_weight=None, normalize=scaled, display_labels=list(self.target_mapper.values()), values_format='d', include_values=True, xticks_rotation=30.0, cmap='Blues', ax=None)
        plt.tick_params(labelsize=13)
        plt.rc("axes", labelsize=12)
        plt.rcParams["font.size"] = 14
        # formatter = "{:d}"
        # formatter = StrMethodFormatter(fmt=formatter)
        # plt.gca().set_major_formatter(formatter=formatter)
        # plt.ticklabel_format(style="plain", useOffset=False)

    def get_accuracy_score(self):
        return "{0:.2f}%".format(100*accuracy_score(self.ytest, self.make_prediction()))

    def get_classification_report(self):
        return classification_report(self.ytest, self.make_prediction())

    def get_confusion_matrix(self):
        return confusion_matrix(self.ytest, self.make_prediction())

    def get_confusion_matrix_params(self):
        tn, fp, fn, tp = confusion_matrix(self.ytest, self.make_prediction()).ravel()
        return 'TruePositive={0}; TrueNegative={1}; FalsePositive (Type1-error)={2}; FalseNegative (Type2-error)={3}.  NOTE: Positive ="{4}"'.format(tp, tn, fp, fn, list(self.target_mapper.values())[1])

    def get_important_features(self, x_df_for_split):
        result = pd.DataFrame(self.model.feature_importances_, index=[map_dataframe_header(pd.DataFrame(self.xtrain),x_df_for_split).columns], columns=["Importance_raw"]).sort_values("Importance_raw", ascending=False)
        result.reset_index(inplace=True)
        result["Importance"] = (np.array(round_num(result["Importance_raw"].values*100, 1)).astype("str"))
        result["Importance"] = result["Importance"]+"%"
        result.columns = ["Feature", "Importance_raw", "Importance"]
        return result[["Feature", "Importance"]]

    def get_score(self):
        return self.model.score(self.xtrain, self.ytrain)


class LogRegression(ClassificationModel):

    from sklearn.linear_model import LogisticRegression
    kwargs = {}
    model = LogisticRegression(solver="lbfgs", **kwargs)

    def __init__(self, xtrain_array, xtest_array, ytrain_array, ytest_array, model=model, target_mapper={}):
        super().__init__(xtrain_array, xtest_array, ytrain_array, ytest_array, model, target_mapper)


class LineRegression(RegressionModel):

    from sklearn.linear_model import LinearRegression
    kwargs = {}
    model = LinearRegression(normalize=True, **kwargs)

    def __init__(self, xtrain_array, xtest_array, ytrain_array, ytest_array, model=model, target_mapper={}):
        super().__init__(xtrain_array, xtest_array, ytrain_array, ytest_array, model, target_mapper)


class RandomForestC(ClassificationModel):

    from sklearn.ensemble import RandomForestClassifier
    kwargs = {}
    model = RandomForestClassifier(n_estimators=1000, **kwargs)

    def __init__(self, xtrain_array, xtest_array, ytrain_array, ytest_array, model=model, target_mapper={}):
        super().__init__(xtrain_array, xtest_array, ytrain_array, ytest_array, model, target_mapper)


class DTreeC(ClassificationModel):

    from sklearn import tree
    kwargs = {}
    model = tree.DecisionTreeClassifier(criterion='gini', **kwargs)

    def __init__(self, xtrain_array, xtest_array, ytrain_array, ytest_array, model=model, target_mapper={}):
        super().__init__(xtrain_array, xtest_array, ytrain_array, ytest_array, model, target_mapper)


def set_nn_params(width=5, kernel_initializer="uniform",
 activation="relu", input_dim=1, batch_size=20, epochs=35,
optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]):
    return {"width": width,
            "kernel_initializer": kernel_initializer,
            "activation": activation, "input_dim": input_dim,
            "optimizer": optimizer,
            "loss": loss, "metrics": metrics,
            "batch_size": batch_size, "epochs": epochs}


print("tensorflow as tf")


class NeuralNetC(ClassificationModel):

    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense

    model = Sequential()
    params = set_nn_params()

    def __init__(self, xtrain_array, xtest_array, ytrain_array, ytest_array, model=model, params=params, target_mapper={}):
        super().__init__(xtrain_array, xtest_array, ytrain_array, ytest_array, model, target_mapper)
        # self.xtrain = xtrain_array
        self.__set_params(width=params["width"], kernel_initializer=params["kernel_initializer"], activation=params["activation"], input_dim=params["input_dim"], optimizer=params["optimizer"], loss=params["loss"], metrics=params["metrics"], batch_size=params["batch_size"], epochs=params["epochs"])

    def set_params(self, width=params["width"], kernel_initializer=params["kernel_initializer"], activation=params["activation"], input_dim=params["input_dim"], optimizer=params["optimizer"], loss=params["loss"], metrics=params["metrics"], batch_size=params["batch_size"], epochs=params["epochs"]):
        return {'width':width, 'kernel_initializer':kernel_initializer,
 'activation':activation, 'input_dim':input_dim, 'optimizer':optimizer,
'loss':loss, 'metrics':metrics, 'batch_size':batch_size, 'epochs':epochs}

    __set_params = set_params

    def get_default_params(self):
        return self.params

    def get_possible_params(self):
        pass

    def add_layer(self, layer_params={"width":params["width"], "kernel_initializer":params["kernel_initializer"], "activation":params["activation"], "input_dim":params["input_dim"]}, **kwargs):
        from tensorflow.keras.layers import Dense
        self.model.add(Dense(layer_params["width"], kernel_initializer=layer_params["kernel_initializer"], activation=layer_params["activation"], input_dim=layer_params["input_dim"], **kwargs))
        return self.model.summary()

    def compile_model(self, optimizer=params["optimizer"], loss=params["loss"], metrics=params["metrics"], **kwargs):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics, **kwargs)

    def fit_model(self, verbose=1, epochs=params["epochs"], batch_size=params["batch_size"], **kwargs):
        global fitted
        fitted = self.model.fit(self.xtrain, self.ytrain, batch_size=batch_size, epochs=epochs, verbose=verbose, validation_data=(self.xtest,self.ytest), **kwargs)

    def get_accuracy_score(self):
        model_eval = self.model.evaluate(self.xtest, self.ytest, verbose=0)
        print("Test Loss:", model_eval[0])
        print("Test Accuracy:", model_eval[1])
        return model_eval

    def plot_score(self, chart_type=["loss", "accuracy"]):
        history_dict = fitted.history
        for item in chart_type:
            if item == "loss":
                plt.cla()
                training_loss = history_dict["loss"]
                validation_loss = history_dict["val_loss"]
                epoch_x = range(1, len(training_loss)+1)

                plot1 = plt.plot(epoch_x, validation_loss, label="Validation Loss")
                plot2 = plt.plot(epoch_x, training_loss, label="Training Loss")

                plt.setp(plot1, linewidth=2.0, marker="+", markersize=10.0)
                plt.setp(plot2, linewidth=2.0, marker="o", markersize=3.0)
                plt.xlabel("Epochs")
                plt.ylabel("Loss")
                plt.title("Loss Chart")
                plt.grid(True)
                plt.legend()
                plt.show()

            else:
                plt.cla()
                training_acc = history_dict["accuracy"]
                validation_acc = history_dict["val_accuracy"]
                epoch_x = range(1, len(training_acc) + 1)

                plot1 = plt.plot(epoch_x, validation_acc, label="Validation Accuracy")
                plot2 = plt.plot(epoch_x, training_acc, label="Training Accuracy")

                plt.setp(plot1, linewidth=2.0, marker="+", markersize=10.0)
                plt.setp(plot2, linewidth=2.0, marker="o", markersize=3.0)
                plt.xlabel("Epochs")
                plt.ylabel("Accuracy")
                plt.title("Accuracy Chart")
                plt.grid(True)
                plt.legend()
                plt.show()


def main():
    pass

if __name__ == '__main__': main()