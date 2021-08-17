# imports
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, SGDRegressor
from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder
from sklearn.model_selection import train_test_split
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.data import get_data, clean_data

import mlflow
from mlflow.tracking import MlflowClient

EXPERIMENT_NAME = "test_experiment"
# Indicate mlflow to log to remote server
MLFLOW_URI = "https://mlflow.lewagon.co/"
#client = MlflowClient()


class Trainer():
    def __init__(self, X="", y="", split=0.25, random_state= 0):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y
        self.split = split # parameter to split the data
        self.randomstate = random_state
        if self.X == "":
            self.df = get_data(nrows=10_000)

    # Spliting function
    def holdout(self):
        #print(self.df.head())
        if self.X == "":
            self.X_train, self.X_test, self.y_train, self.y_test = \
                train_test_split(self.df.drop(columns="fare_amount"), self.df.fare_amount,
                                    test_size=self.split)
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = \
                train_test_split(self.X, self.y, test_size=self.split)

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        '''returns a pipelined model'''
        dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer()),
            ('stdscaler', StandardScaler())
        ])
        #print("ok")
        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])
        #print("ok")
        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude",
                                    "pickup_longitude",
                                    'dropoff_latitude',
                                    'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")
        #print("ok")
        pipe = Pipeline([
            ('preproc', preproc_pipe),
            ('linear_model', SGDRegressor())
        ])
        #print("ok")
        #print(pipe)
        self.pipeline = pipe
        #return pipe
        #return pipe

    def run(self):
        """set and train the pipeline"""
        self.holdout()
        self.set_pipeline()
        self.pipeline.fit(self.X_train, self.y_train)

    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)

    def evaluate(self):
        """evaluates the pipeline on df_test and return the RMSE"""
        self.run()
        return compute_rmse(self.pipeline.predict(self.X_test), self.y_test)

if __name__ == "__main__":
    df = get_data()
    print("1")
    df = clean_data(df)
    print("2")
    X = df.drop(columns="fare_amount")
    print("3")
    y = df.fare_amount
    print("4")
    trainer = Trainer(X, y)
    print("5")
    trainer.holdout()
    print("6")
    #trainer.set_pipeline()
    # train
    trainer.run()
    print("7")
    # evaluate
    print(trainer.evaluate())
    print('TODO')
