import pandas as pd
import numpy as np

from cbcdb import DBManager
from dotenv import load_dotenv, find_dotenv
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import  mean_squared_error, explained_variance_score, r2_score
import statsmodels.api as sm
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import mlflow
mlflow.tensorflow.autolog(True)
load_dotenv(find_dotenv())


class CustomerRetention():
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
            create temporary table consecutive_days as (
                 select uid
                 , datetime_
                 , rank_group
                 , visit_number
                 , max(visit_number) over (partition by uid) as max_num_visit
                 , case when (max_num_visit) > 1 then 1
                    else 0 end as visit_more_than_once
                from (
                     select uid
                          , datetime_
                          , rank_group
                          , rank() over (partition by uid order by rank_group asc) as visit_number
                     from (
                              SELECT uid
                                   , datetime                                                                      as datetime_
                                   , dateadd(day, -rank() OVER (partition by uid ORDER BY datetime), datetime)     AS rank_group
                              FROM (SELECT DISTINCT
                                                   t.location_id || '_' || t.animal_id as uid
                                                  , trunc(t.datetime_date) as datetime
                                    from bi.transactions t
                                             inner join bi.animals a
                                                        on a.ezyvet_id = t.animal_id
                                                            and a.location_id = t.location_id
                                    order by 1, 2))));
                    select f1.uid
                         , f1.breed
                         , f1.ani_age
                         , f1.date
                         , f1.weight
                         , f1.name
                         , f1.type
                         , f1.tracking_level
                         , f1.product_group
                         , f1.visit_number
                         , f1.visit_more_than_once
                         , f1.max_num_visit
                         , f1.first_visit_spend
                         , f1.total_future_spend
                        from(
                        select f.uid
                               , f.breed
                               , f.ani_age
                               , f.date
                               , f.weight
                               , f.name
                               , f.type
                               , f.tracking_level
                               , f.product_group
                               , cd.visit_number
                               , cd.visit_more_than_once
                               , cd.max_num_visit
                               , sum(case when cd.visit_number != 1 then f.revenue else 0 end)
                                    over (partition by f.uid) as total_future_spend
                                , sum(case when cd.visit_number = 1 then f.revenue else 0 end)
                                    over (partition by f.uid) as first_visit_spend
                                from (select t.location_id || '_' || t.animal_id as uid
                                     , a.breed
                                     , max(
                                      date_diff('years', timestamp 'epoch' + a.date_of_birth * interval '1 second',
                                                current_date))                                            as ani_age
                                     , trunc(t.datetime_date)                                            as date
                                     , a.weight
                                     , p.name
                                     , p.type
                                     , p.tracking_level
                                     , p.product_group
                                     , t.revenue                                                          as revenue
                                     , dense_rank() over (partition by t.location_id || '_' || t.animal_id  order by trunc(t.datetime_date)  asc) as rank_
                                    from bi.transactions t
                                         inner join bi.products p
                                                    on t.product_id = p.ezyvet_id
                                                        and t.location_id = p.location_id
                                         inner join bi.animals a
                                                    on a.id = t.animal_id
                                         left join bi.contacts c
                                                   on a.contact_id = c.ezyvet_id
                                                       and t.location_id = c.location_id
                                     --and p.is_medical = 1
                                where p.name not like '%Subscri%'
                                  and p.product_group != 'Surgical Services'
                                  --and a.breed != '0.0'
                                group by 1, 2, 4, 5, 6, 7, 8, 9, 10) f
                    
                                   left join consecutive_days cd
                                             on f.uid = cd.uid
                                                 and f.date = cd.datetime_) f1
                    where f1.visit_number = 1
                    order by 1, 4;"""
        try:
            self.df = pd.read_csv('data/data.csv', index=False)
        except:
            self.df = self.db.get_sql_dataframe(sql)
            self.df.to_csv('data/data.csv')

    def feature_eng(self):
        self.df = self.df[((self.df.weight != 0.0) | (self.df.breed != '0.0') | (self.df.total_future_spend > 10000))]

        self.df.reset_index(drop=True,inplace=True)
        df_ = self.df[self.df.visit_number == 1][
            ['uid', 'ani_age', 'weight', 'first_visit_spend', 'total_future_spend', 'product_group', 'breed']]
        df_product_group = pd.get_dummies(self.df.product_group)
        df_breed = pd.get_dummies(self.df.breed)
        df_ = pd.concat([df_[['uid', 'ani_age', 'weight', 'first_visit_spend', 'total_future_spend']],
                         df_breed,
                         df_product_group], axis=1)

        columns_to_sum = list(df_.columns)
        for i in ['uid', 'weight', 'ani_age', 'first_visit_spend', 'total_future_spend']:
            columns_to_sum.remove(i)

        df_final = df_.groupby(['uid', 'weight', 'ani_age','first_visit_spend', 'total_future_spend'], as_index=False)[columns_to_sum].sum()


        df_final['total_future_spend'] = df_final['total_future_spend'].apply(lambda x: 0 if x < 0 else x)
        bins =[0,100,200,300,99999]
        labels = [0,1,2,3]
        df_final['total_future_spend_bin'] = pd.cut(df_final['total_future_spend'],bins=bins, include_lowest=True, labels=labels)

        return df_final

    def split_train_test(self, df, label: str = 'total_future_spend'):
        final_columns = list(df.columns)
        for i in ['uid', 'total_future_spend', 'total_future_spend_bin']:
            final_columns.remove(i)

        X = df[final_columns]
        y = df[label].apply(lambda x: int(x))
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=.2,
                                                            random_state=42)
        return X_train, X_test, y_train, y_test

    def model_fit(self, X_train, y_train):
        self.model = Sequential()
        self.model.add(Dense(256, input_dim=X_train.shape[1], activation='relu'))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(1, activation='linear'))

        self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
        es = EarlyStopping(monitor='mse', mode='min', verbose=1, patience=3)
        self.model.fit(X_train,
                  y_train,
                  epochs=100,
                  batch_size=10,
                  validation_split=.2,
                  callbacks=[es])

    def predict(self, X_test):
        # Predict
        y_pred = self.model.predict(X_test)
        return y_pred

    def mlflow_metrics(self, y_test, y_pred):
        # export_path = mlflow.active_run().info.artifact_uri
        # builder = tf.saved_model.builder.SavedModelBuilder(export_path)
        # # Save the model
        # builder.save(as_text=True)
        #
        # # log the model
        # mlflow.log_artifacts(export_path, "model")
        # mlflow.tensorflow.log_model(tf_saved_model_dir=export_path,
        #                             artifact_path="model")

        # store metrics
        r2 = r2_score(y_test, y_pred)
        exp_var = explained_variance_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        mlflow.log_metric("R2", r2)
        mlflow.log_metric("Explained Variance Score", exp_var)
        mlflow.log_metric("Mean Squared Error", mse)
        mlflow.log_metric("Root Mean Squared Error", rmse)


if __name__ == '__main__':
    mlflow.set_tracking_uri('/Users/adhamsuliman/Documents/cbc/pwa/pwa-ds/mlruns')
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
        cr = CustomerRetention()
        # ep.validate_model_pred()
        cr.read_in_latest()
        df = cr.feature_eng()
        X_train, X_test, y_train, y_test = cr.split_train_test(df)
        cr.model_fit(X_train, y_train)
        y_pred = cr.predict(X_test)
        cr.mlflow_metrics(y_test, y_pred)