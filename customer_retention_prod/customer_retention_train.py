import pandas as pd
import numpy as np

from cbcdb import DBManager
from dotenv import load_dotenv, find_dotenv
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, explained_variance_score, r2_score, f1_score, recall_score, \
    precision_score
from sklearn.utils import class_weight
import statsmodels.api as sm
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import mlflow

mlflow.tensorflow.autolog(True)
load_dotenv(find_dotenv())


class CustomerRetention():
    def __init__(self):
        self.db = DBManager()

    def read_in_latest(self):
        sql = """    
            -- medical visit
            -- pull doctor's notes
                -- look into type_id
                -- (look for trauma and acute) probably not coming back
                -- routines, vaccines, nutures will probably come back
             -- wellness plan into a binary indicator
             -- look into conversions for 2nd to 3rd etc, not only 1st conversion.

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

            create temporary table wellness as (
                    select
                          wm.location_id || '_' || wm.animal_id as uid
                         , date(datetime_start_date) as datetime_
                         , wm.wellness_plan as wellness_plan_num
                         , DATEDIFF(MONTH, wm.datetime_start_date, CURRENT_DATE) as months_a_member
                         , wp.name                                               as wellness_plan
                    from bi.wellness_membership wm
                             left join bi.wellness_plans wp
                                 on wp.location_id = wm.location_id
                                        and wp.ezyvet_id = wm.wellness_plan
                             left join bi.animals a
                                 on a.location_id = wm.location_id
                                        and a.ezyvet_id = wm.animal_id);
                    -- where wp.active = 1
                    --  and wm.status = 'Active');

            select f1.uid
                     , f1.breed
                     , f1.ani_age
                     , f1.date
                     , f1.weight
                     , f1.is_medical
                     , f1.product_group
                     , f1.type_id
                     , f1.wellness_plan
                     , f1.months_a_member
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
                           , f.is_medical
                           --, f.tracking_level
                           , f.product_group
                           , f.type_id
                           , cd.visit_number
                           , cd.visit_more_than_once
                           , cd.max_num_visit
                           , w.wellness_plan
                           , w.months_a_member
                           , sum(case when cd.visit_number != 1 then f.revenue else 0 end)
                                over (partition by f.uid) as total_future_spend
                            , sum(case when cd.visit_number = 1 then f.revenue else 0 end)
                                over (partition by f.uid) as first_visit_spend
                            from (
                                select t.location_id || '_' || t.animal_id as uid
                                 , a.breed
                                 , max(
                                  date_diff('years', timestamp 'epoch' + a.date_of_birth * interval '1 second',
                                            current_date))                                            as ani_age
                                 , trunc(t.datetime_date)                                            as date
                                 , a.weight
                                 , p.is_medical
                                 , p.product_group
                                 , case when apt.type_id like 'Grooming%' then 'groom'
                                        when apt.type_id like '%Neuter%' then 'neurtering'
                                        when apt.type_id like '%ental%' then 'dental'
                                        else apt.type_id end as type_id
                                 --, p.name
                                 --, p.type
                                 --, p.tracking_level
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
                                    left join bi.appointments apt
                                               on a.contact_id = apt.ezyvet_id
                                                   and t.location_id = apt.location_id
                            --where p.name not like '%Subscri%'
                            --  and p.product_group != 'Surgical Services'
                              --and a.breed != '0.0'
                            group by 1, 2, 4, 5, 6, 7, 8,9) f
            
                               left join consecutive_days cd
                                         on f.uid = cd.uid
                                             and f.date = cd.datetime_
                                left join wellness w
                                        on f.uid = w.uid
                                            and f.date = w.datetime_) f1
                where f1.visit_number = 1
                order by 1, 4;"""
        self.df = self.db.get_sql_dataframe(sql)
        self.df.to_csv('data/data.csv')

    def feature_eng(self):
        print(f"Number of unique id\'s priot to filtering 0 from weight and breed: {self.df.uid.nunique()}")
        (self.df.weight == 0).sum()
        (self.df.breed == '0.0').sum()
        self.df = self.df[((self.df.weight != 0) & (self.df.breed != '0.0'))]
        print(f"Number of unique id\'s : {self.df.uid.nunique()}")
        print(
            f"There are {self.df[self.df.total_future_spend > 5000]['uid'].nunique()} patients who have spent more than 5k")
        print(
            f"There are {self.df[self.df.total_future_spend < 0]['uid'].nunique()} patients who have somehow spent less than $0")
        self.df['total_future_spend'] = self.df.total_future_spend.apply(lambda x: 5000 if x > 5000 else x)
        self.df['total_future_spend'] = self.df['total_future_spend'].apply(lambda x: 0 if x < 0 else x)





        #self.df = self.df[((self.df.ani_age.notnull()) & (self.df.weight.notnull()))]
        self.df['ani_age'] = self.df['ani_age'].fillna((self.df['ani_age'].mean()))
        self.df['weight'] = self.df['weight'].fillna((self.df['weight'].mean()))
        self.df['weight'] = self.df['weight'].apply(lambda x: self.df['weight'].mean() if x == 0 else x)

        self.df.reset_index(drop=True, inplace=True)
        df_ = self.df[self.df.visit_number == 1][['uid', 'ani_age', 'weight', 'is_medical', 'product_group', 'type_id', 'wellness_plan', 'first_visit_spend','total_future_spend']]
        df_main = df_.groupby(['uid', 'ani_age', 'weight', 'first_visit_spend', 'total_future_spend'], as_index=False)['is_medical'].max()
        df_product_group = pd.get_dummies(df_.product_group)
        df_type = pd.get_dummies(df_.type_id)
        df_ = pd.concat([df_[['uid']],
                         df_type,
                         df_product_group], axis=1)  # .fillna(0)
        df_ = df_.groupby(['uid']).sum()
        df_final = df_main.merge(df_,on='uid')

        bins = [0, 100, 200, 300, 1000, 99999]
        self.labels = [0, 1, 2, 3, 4]
        df_final['total_future_spend_bin'] = pd.cut(df_final['total_future_spend'], bins=bins, include_lowest=True,labels=self.labels)
        print(f"Value Counts for labels: {df_final['total_future_spend_bin'].value_counts()}")
        return df_final

    def split_train_test(self, df, label: str = 'total_future_spend_bin'):
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

    def model_fit(self, X_train, y_train, df):
        self.model = Sequential()
        self.model.add(Dense(512, input_dim=X_train.shape[1], activation='relu'))
        self.model.add(Dense(256, input_dim=X_train.shape[1], activation='relu'))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(len(self.labels), activation='softmax'))

        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)

        if isinstance(df, pd.DataFrame):
            class_weights = class_weight.compute_class_weight('balanced',
                                                              np.unique(self.labels),
                                                              list(df['total_future_spend_bin'].values))
            class_weights = dict(enumerate(class_weights))

            self.model.fit(X_train,
                           y_train,
                           epochs=100,
                           batch_size=10,
                           validation_split=.2,
                           callbacks=[es],
                           class_weight=class_weights)
        else:
            self.model.fit(X_train,
                           y_train,
                           epochs=100,
                           batch_size=10,
                           validation_split=.2,
                           callbacks=[es])

    def predict(self, X_test: np.array):
        # Predict
        y_pred = self.model.predict(X_test)
        y_pred = np.argmax(y_pred, axis=-1)

        return y_pred

    def mlflow_metrics(self, y_test, y_pred, linear: bool = False):
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
        if linear:
            r2 = r2_score(y_test, y_pred)
            exp_var = explained_variance_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)

            mlflow.log_metric("R2", r2)
            mlflow.log_metric("Explained Variance Score", exp_var)
            mlflow.log_metric("Mean Squared Error", mse)
            mlflow.log_metric("Root Mean Squared Error", rmse)

        else:
            f1_score_0 = f1_score(y_test, y_pred, labels=[0, 1, 2, 3, 4], average="weighted")
            recall_score_0 = recall_score(y_test, y_pred, labels=[0, 1, 2, 3, 4], average="weighted")
            precision_score_0 = precision_score(y_test, y_pred, labels=[0, 1, 2, 3, 4], average="weighted")

            f1_score_ = f1_score(y_test, y_pred, labels=[1, 2, 3, 4], average="weighted")
            recall_score_ = recall_score(y_test, y_pred, labels=[1, 2, 3, 4], average="weighted")
            precision_score_ = precision_score(y_test, y_pred, labels=[1, 2, 3, 4], average="weighted")

            y_test_ = [1 if x > 0 else 0 for x in y_test]
            y_pred_ = [1 if x > 0 else 0 for x in y_pred]
            f1_score_binary = f1_score(y_test_, y_pred_, average="weighted")
            recall_score_binary = recall_score(y_test_, y_pred_, average="weighted")
            precision_score_binary = precision_score(y_test_, y_pred_, average="weighted")

            mlflow.log_metric("F1", f1_score_0)
            mlflow.log_metric("Recall", recall_score_0)
            mlflow.log_metric("Precision", precision_score_0)

            mlflow.log_metric("F1 W/O 0", f1_score_)
            mlflow.log_metric("Recall W/O 0", recall_score_)
            mlflow.log_metric("Precision W/O 0", precision_score_)

            mlflow.log_metric("F1 Binary", f1_score_binary)
            mlflow.log_metric("Recall Binary", recall_score_binary)
            mlflow.log_metric("Precision Binary", precision_score_binary)


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
    with mlflow.start_run(run_name=f'W/imputation'):
        cr = CustomerRetention()
        # ep.validate_model_pred()
        cr.read_in_latest()
        df = cr.feature_eng()
        X_train, X_test, y_train, y_test = cr.split_train_test(df)
        cr.model_fit(X_train, y_train, df)
        y_pred = cr.predict(X_test)
        cr.mlflow_metrics(y_test, y_pred)
