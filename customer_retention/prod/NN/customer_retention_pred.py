import pandas as pd
import numpy as np

from cbcdb import DBManager
from dotenv import load_dotenv, find_dotenv
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
import statsmodels.api as sm
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import mlflow
load_dotenv(find_dotenv())


class CustomerRetentionPred():
    def __init__(self):
        self.db = DBManager()

    @staticmethod
    def Find_Optimal_Cutoff(target, predicted):
        """ Find the optimal probability cutoff point for a classification model related to event rate
        Parameters
        ----------
        target : Matrix with dependent or target data, where rows are observations

        predicted : Matrix with predicted data, where rows are observations

        Returns
        -------
        list type, with optimal cutoff value

        """
        fpr, tpr, threshold = roc_curve(target, predicted)
        i = np.arange(len(tpr))
        roc = pd.DataFrame({'tf': pd.Series(tpr - (1 - fpr), index=i), 'threshold': pd.Series(threshold, index=i)})
        roc_t = roc.iloc[(roc.tf - 0).abs().argsort()[:1]]

        return list(roc_t['threshold'])

    def read_in_latest(self):
        sql = """
            select t.animal_id
             , a.species
             , max(date_diff('years', timestamp 'epoch' + a.date_of_birth * interval '1 second', current_date)) as ani_age
             , trunc(t.datetime_date)                                                                           as datetime
             , a.weight
             , p.name
             , p.type
             , p.tracking_level
             , p.product_group
             , case
                   when a.id in (select
                                distinct x.animal_id
                                from (
                                    select t.animal_id
                                            , trunc(t.datetime_date)  as date
                                            , sum(t.revenue)          as revenues
                                            , count(1)  over (partition by t.animal_id)              as counts
                                       from bi.transactions t
                                                left join bi.divisions d
                                                          on d.division_id = t.division_id
                                                inner join bi.animals a
                                                           on a.id = t.animal_id
                                                            --and a.location_id = t.location_id
                                                inner join bi.products p
                                                           on t.product_id = p.ezyvet_id
                                                               and t.location_id = p.location_id
                                                               and p.is_medical = 1
                                       where is_dead = 0
                                         and a.active = 1
                                       group by 1, 2
                                       having  revenues > 100) x
                                where x.counts > 1)
                    then 1 else 0 end                                                                        as visit_more_than_once
                 , dense_rank() over (partition by t.animal_id order by datetime asc)                        as rank_
                 , sum(t.revenue)                                                                            as revenue
            from bi.transactions t
                     inner join bi.products p
                                on t.product_id = p.ezyvet_id
                                    and t.location_id = p.location_id
                                    and p.is_medical = 1
                     inner join bi.animals a
                                on a.id = t.animal_id
                     left join bi.contacts c
                               on a.contact_id = c.ezyvet_id
                                   and t.location_id = c.location_id
            where 
                p.name not like '%Subscri%'
              --and p.product_group != 'Surgical Services'
            group by 1, 2, 4, 5, 6, 7, 8, 9, 10;"""
        self.df = self.db.get_sql_dataframe(sql)

    def feature_eng(self):
        self.df = self.df[((self.df.weight != 0.0) | (self.df.breed != '0.0'))]

        self.df.reset_index(drop=True,inplace=True)
        df_ = self.df[self.df.visit_number == 1][
            ['uid', 'ani_age', 'weight', 'revenue', 'total_future_spend', 'product_group', 'breed']]
        df_product_group = pd.get_dummies(self.df.product_group)
        df_breed = pd.get_dummies(self.df.breed)
        df_ = pd.concat([df_[['uid', 'ani_age', 'weight', 'revenue', 'total_future_spend']],
                         df_breed,
                         df_product_group], axis=1)

        df_final = df_.groupby(['uid', 'ani_age', 'weight', 'revenue', 'total_future_spend','breed'], as_index=False)['Canine (Dog)',
        'Feline (Cat)', 'Dentistry & Oral Surgery', 'Diagnostics',
        'Laboratory - Antech', 'Laboratory - In-house', 'Medications - Oral',
        'Parasite Control', 'Professional Services', 'Promotions', 'Vaccinations',
        'Wellness Plan Fees'].sum()
        return df_


    def split_train_test(self, df):
        final_columns = list(df.columns)
        for i in ['uid', 'total_future_spend', 'total_future_spend_bin']:
            final_columns.remove(i)

        X = df[final_columns]
        y = df['total_future_spend'].apply(lambda x: int(x))
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=.2,
                                                            random_state=42)
        return X_train, X_test, y_train, y_test

    def model_fit(self):
        res = sm.GLM(y_train, X_train, family=sm.families.Binomial()).fit()


class MlflowCallback(tf.keras.callbacks.Callback):

    # This function will be called after each epoch.
    def on_epoch_end(self, epoch, logs=None):
        if not logs:
            return
        # Log the metrics from Keras to MLflow
        mlflow.log_metric("loss", logs["loss"], step=epoch)
        mlflow.log_metric("acc", logs["acc"], step=epoch)

        # This function will be called after training completes.
        def on_train_end(self, logs=None):
            mlflow.log_param('num_layers', len(self.model.layers))
            mlflow.log_param('optimizer_name', type(self.model.optimizer).__name__)


if __name__ == '__main__':
    mlflow.set_tracking_uri(os.environ.get('MLFLOW_TRACKING_URI'))
    # mlflow.create_experiment(name='Employee Churn')
    mlflow.set_experiment('Cust 1.5 year value')

    # names = ['big_layer', 'smaller_layer']
    # lstm_layer_1_sizes = [256, 128]
    # lstm_layer_2_sizes = [128, 64]
    #
    # parameters_list = zip(names, lstm_layer_1_sizes, lstm_layer_2_sizes)
    # for n, l1, l2 in parameters_list:
    #     # start mlflow run
    with mlflow.start_run(run_name=f'8_02_21'):
        cr = CustomerRetentionPred(read=True)
        # ep.validate_model_pred()
        cr.read_in_latest()
        X_train, X_test, y_train, y_test, test_unique_ID = cr.feature_eng()
        cr.create_model_and_fit(X_train, y_train, l1, l2)
        y_pred = cr.predict(X_test, y_test)
        cr.mlflow_metrics(X_test, y_test, y_pred)
