import pandas as pd
import numpy as np


from cbcdb import DBManager
from dotenv import load_dotenv, find_dotenv
import matplotlib.pyplot as plt
import mlflow
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, explained_variance_score, r2_score, f1_score, recall_score, \
    precision_score, classification_report
import sys
sys.path.append(os.getcwd())
from customer_retention.util.breed_identifier import BreedIdentifier

import os
import xgboost as xgb


load_dotenv(find_dotenv())


class CustomerRetentionTrain():
    def __init__(self):
        self.azure_key = os.environ.get('AZURE_KEY')

    def start(self,
              objective='multi:softprob'):

        db = DBManager()
        # Read in the latest data
        df = self.read_in_latest(db)

        # Return dog Breed
        df = self.dog_breed(df, db)

        # Begin feature engineering
        df = self.feature_eng(df)

        # Label Engineering
        self.label_eng(df, objective)

        # Split into train and test
        X_train, X_test, y_train, y_test = self.split_train_test(df)

        # Fit the model
        model, y_pred = self.model_fit(X_train, X_test, y_train, y_test, objective=objective)

        # Store metrics on mlflow
        self.mlflow_metrics(model, y_test, y_pred)

    def read_in_latest(self, db: DBManager, export: bool = True):
        sql = """    
        create temporary table consecutive_days as (
            select uid
                 , datetime_
                 , rank_group
                 , visit_number
                 , max(visit_number) over (partition by uid) as max_num_visit
                 , case
                       when (max_num_visit) > 1 then 1
                       else 0 end                            as visit_more_than_once
            from (
                     select uid
                          , datetime_
                          , rank_group
                          , dense_rank() over (partition by uid order by rank_group asc) as visit_number
                     from (
                              SELECT uid
                                   , datetime                                                                  as datetime_
                                   , dateadd(day, -rank() OVER (partition by uid ORDER BY datetime), datetime) AS rank_group
                              FROM (SELECT DISTINCT t.location_id || '_' || t.animal_id as uid
                                                  , trunc(t.datetime_date)              as datetime
                                                    , sum(t.revenue) as revenue_
                                    from bi.transactions t
                                             inner join bi.animals a
                                                        on a.ezyvet_id = t.animal_id
                                                            and a.location_id = t.location_id
                                    group by 1,2
                              having revenue_ > 0
                                    order by 1, 2))));
        
        create temporary table wellness as (
            select wm.location_id || '_' || wm.animal_id                 as uid
                 , date(datetime_start_date)                             as datetime_
                 , wm.wellness_plan                                      as wellness_plan_num
                 , DATEDIFF(MONTH, wm.datetime_start_date, CURRENT_DATE) as months_a_member
                 , wp.name                                               as wellness_plan
            from pwa_bi.bi.wellness_membership wm
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
             , f1.product_name
             ,  sum(case when f1.product_name like 'First Day Daycare Fre%' then 1 else 0 end)
                     over (partition by f1.uid)  as product_count
             --, f1.type_id
             , f1.wellness_plan
             --, f1.months_a_member
             , f1.visit_number
             , f1.visit_more_than_once
             , f1.max_num_visit
             , f1.first_visit_spend
             , f1.total_future_spend
        from (
                     select f.uid
                      , f.breed
                      , f.ani_age
                      , f.date
                      , f.weight
                      , f.is_medical
                      --, f.tracking_level
                      , f.product_group
                      , f.product_name
                      --, f.type_id
                      , cd.visit_number
                      , cd.visit_more_than_once
                      , cd.max_num_visit
                      , f.revenue
                      ,  sum(case when w.wellness_plan is null then 0 else 1 end)
                        over (partition by f.uid)  as wellness_plan
                      --, w.months_a_member
                      , sum(case when cd.visit_number != 1 then f.revenue else 0 end)
                        over (partition by f.uid) as total_future_spend
                      , sum(case when cd.visit_number = 1 then f.revenue else 0 end)
                        over (partition by f.uid) as first_visit_spend
                 from (

                          select t.location_id || '_' || t.animal_id                                                         as uid
                               , a.breed
                               , max(
                                  date_diff('years', timestamp 'epoch' + a.date_of_birth * interval '1 second',
                                            current_date))                                                                   as ani_age
                               , case when trunc(t.datetime_date) - min(trunc(t.datetime_date)) over (partition by t.location_id || '_' || t.animal_id) > 548 then 0
                                   else 1 end as less_than_1_5_yeras
                                , case when current_date - min(trunc(t.datetime_date)) over (partition by t.location_id || '_' || t.animal_id) <  270 then 0
                                   else 1 end as recent_patient
                               , trunc(t.datetime_date)                                                                      as date
                               , a.weight
                               , p.is_medical
                                , case when  p.product_group like 'Retail%' then 'Retail'
                                        when p.product_group in ('Boarding','Daycare','Daycare Packages') then 'Boarding'
                                        when p.product_group like 'Medication%' then 'Medication'
                                        when p.product_group like  ('Referrals%') then 'Referrals'
                                        when p.product_group in ('to print','To Print','Administration') then 'admin'
                                        else p.product_group end as product_group
                               --, p.name
                               --, p.type
                               --, p.tracking_level
                               , t.product_name
                               , sum(t.revenue)                                                                                   as revenue
                               , dense_rank()
                                 over (partition by t.location_id || '_' || t.animal_id order by trunc(t.datetime_date) asc) as rank_
                          from bi.transactions t
                                   --inner join bi.products p
                                   left join bi.products p
                                              on t.product_id = p.ezyvet_id
                                                  and t.location_id = p.location_id
                                   inner join bi.animals a
                                              on a.id = t.animal_id
                                   left join bi.contacts c
                                             on a.contact_id = c.ezyvet_id
                                                 and t.location_id = c.location_id
                                --where t.product_name like 'First Day Daycare Free%'
                                -- type_id not in ('Cancellation')
                                -- p.name not like '%Subscri%'
                                --  and p.product_group != 'Surgical Services'
                                -- and a.breed != '0.0'
                          group by 1, 2, 6, 7, 8, 9, 10
                     ) f
                  inner join consecutive_days cd
                            on f.uid = cd.uid
                                and f.date = cd.datetime_
                  left join wellness w
                            on f.uid = w.uid
                                and f.date = w.datetime_
                    where less_than_1_5_yeras = 1
                            and recent_patient = 1) f1
        where f1.visit_number = 1
        order by 1, 4;
        """
        df = db.get_sql_dataframe(sql)

        return df

    def dog_breed(self, df, db):
        df_breed = db.get_sql_dataframe("select * from bi.breeds")
        breed = set(df.breed.unique().tolist())
        breed_orig = set(df_breed.breed.unique().tolist())

        new_breeds = list(breed - breed_orig)
        if ((len(new_breeds) > 0) and (new_breeds[0] is not None)):
            bi = BreedIdentifier()
            bi.start(new_breeds)
            return self.dog_breed(df, db)
        else:
            df1 = df.merge(df_breed, on='breed')
            return df1

    @staticmethod
    def feature_eng(df: pd.DataFrame) -> pd.DataFrame:
        print(
            f"Out of {df.uid.nunique()}, there are {df[((df.weight.isnull()) | (df.weight == 0.0))]['uid'].nunique()} animals with null weight)")
        print(f"Out of {df.uid.nunique()}, there are {df[df.ani_age.isnull()]['uid'].nunique()} animals with null age)")

        print(f"Number of unique id\'s : {df.uid.nunique()}")
        print(f"There are {df[df.total_future_spend > 5000]['uid'].nunique()} patients who have spent more than 5k")
        print(
            f"There are {df[df.total_future_spend < 0]['uid'].nunique()} patients who have somehow spent less than $0")
        df.reset_index(drop=True, inplace=True)
        df_ = df[['uid', 'ani_age', 'weight', 'product_group', 'breed_group', 'tier', 'is_medical',
                  'wellness_plan', 'first_visit_spend', 'total_future_spend']]

        # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
        # # Create categorical df. Number of rows should equate to the number of unique uid
        # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

        cat_features = ['wellness_plan', 'breed_group', 'tier']
        df_cat = df_[['uid'] + cat_features].drop_duplicates()
        # df_product_group = pd.get_dummies(df_.product_group)
        df_breed_group = pd.get_dummies(df_cat.breed_group)
        df_tier = pd.get_dummies(df_cat.tier)

        df_cat = pd.concat([df_cat[['uid']],
                            # df_cat.product_group,
                            df_breed_group,
                            df_tier], axis=1)  #
        df_categories = df_cat.groupby(['uid'], as_index=False).max()

        # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
        # Create standardized df. Number of rows should equate to the number of unique uid
        # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
        cont_features = ['ani_age', 'weight', 'breed_group', 'is_medical', 'wellness_plan', 'first_visit_spend',
                         'total_future_spend']
        df_cont = df_[['uid'] + cont_features].drop_duplicates()

        # look into a binary indicator for weight
        df_cont['ani_age'] = df_cont['ani_age'].fillna((df_cont['ani_age'].mean()))
        df_cont['weight'] = df_cont['weight'].replace(0, np.nan)
        df_cont['weight'] = df_cont.groupby(['ani_age', 'breed_group'])['weight'].transform(
            lambda x: x.fillna(x.mean()))
        df_cont['weight'] = df_cont.groupby(['breed_group'])['weight'].transform(lambda x: x.fillna(x.mean()))

        df_cont = df_cont.groupby(['uid', 'ani_age', 'weight', 'first_visit_spend', 'total_future_spend'],
                                  as_index=False).agg(
            is_medical_max=('is_medical', 'max'),
            is_medical_count=('is_medical', 'count'),
            wellness_plan_max=('wellness_plan', 'max'),
            wellness_plan_count=('wellness_plan', 'count')
        )

        df_final = df_cont.merge(df_categories, on='uid')
        df_final['total_future_spend'] = df_final.total_future_spend.apply(lambda x: 5000 if x > 5000 else x)
        df_final['total_future_spend'] = df_final['total_future_spend'].apply(lambda x: 0 if x < 0 else x)
        return df_final
        # self.df_final = self.df_final[((self.df_final.ani_age.notnull()) & (self.df_final.weight.notnull()))]
        # self.df_final['weight'] = self.df_final['weight'].fillna((self.df_final['weight'].mean()))
        # self.df_final['weight'] = self.df_final['weight'].apply(lambda x: self.df_final['weight'].mean() if x == 0 else x)

    @staticmethod
    def label_eng(df: pd.DataFrame, objective: str):
        # Get dummies on tier and breed group
        if objective == 'binary:hinge':
            bins = [0, 1, 99999]
            labels = [0, 1]
            df['total_future_spend_bin'] = pd.cut(df['total_future_spend'], bins=bins, include_lowest=True,
                                                        labels=labels)
        else:
            bins = [0, 1, 200, 350, 500, 1000, 99999]
            labels = [0, 1, 2, 3, 4, 5]
            df['total_future_spend_bin'] = pd.cut(df['total_future_spend'], bins=bins, include_lowest=True,
                                                        labels=labels)


    @staticmethod
    def split_train_test(df, label: str = 'total_future_spend_bin'):
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

    def model_fit(self, X_train, X_test, y_train, y_test, objective):
        dtrain = xgb.DMatrix(X_train, label=y_train, missing=-999.0)
        dtest = xgb.DMatrix(X_test, label=y_test, missing=-999.0)
        evallist = [(dtest, 'eval'), (dtrain, 'train')]
        param = {'max_depth': 5,
                 'objective': objective,
                 'num_class': 6,
                 'nthread': 4,
                 'eval_metric': ['merror','mlogloss','auc']}

        num_round = 20
        bst = xgb.train(param, dtrain, num_round, evallist)
        y_pred = bst.predict(dtest)
        y_pred = np.argmax(y_pred, axis=-1)
        return bst, y_pred

    def mlflow_metrics(self, bst, y_test, y_pred):
        ax = xgb.plot_importance(bst)
        ax.figure.tight_layout()
        ax.figure.savefig('customer_retention/prod/artifacts/feature_importance.png')
        mlflow.log_artifact("customer_retention/prod/artifacts/feature_importance.png")
        plt.close()

        mlflow.xgboost.log_model(bst,
                                "xgboost",
                                 conda_env=mlflow.xgboost.get_default_conda_env())

        # store metrics
        f1_score_0 = f1_score(y_test, y_pred, labels=[0, 1, 2, 3, 4, 5], average="weighted")
        recall_score_0 = recall_score(y_test, y_pred, labels=[0, 1, 2, 3, 4, 5], average="weighted")
        precision_score_0 = precision_score(y_test, y_pred, labels=[0, 1, 2, 3, 4, 5], average="weighted")

        f1_score_ = f1_score(y_test, y_pred, labels=[1, 2, 3, 4, 5], average="weighted")
        recall_score_ = recall_score(y_test, y_pred, labels=[1, 2, 3, 4, 5], average="weighted")
        precision_score_ = precision_score(y_test, y_pred, labels=[1, 2, 3, 4, 5], average="weighted")

        y_test_ = [1 if x > 0 else 0 for x in y_test]
        y_pred_ = [1 if x > 0 else 0 for x in y_pred]
        f1_score_binary = f1_score(y_test_, y_pred_, average="weighted")
        recall_score_binary = recall_score(y_test_, y_pred_, average="weighted")
        precision_score_binary = precision_score(y_test_, y_pred_, average="weighted")

        mlflow.log_metric("F1", f1_score_0)
        mlflow.log_metric("Recall", recall_score_0)
        mlflow.log_metric("Precision", precision_score_0)

        mlflow.log_metric("F1_W/O_0", f1_score_)
        mlflow.log_metric("Recall_W/O_0", recall_score_)
        mlflow.log_metric("Precision_W/O_0", precision_score_)

        mlflow.log_metric("F1_Binary", f1_score_binary)
        mlflow.log_metric("Recall_Binary", recall_score_binary)
        mlflow.log_metric("Precision_Binary", precision_score_binary)


if __name__ == '__main__':
    mlflow.set_tracking_uri(os.environ.get('MLFLOW__CORE__SQL_ALCHEMY_CONN'))
    #mlflow.create_experiment('Future Cust Value', os.environ.get('EXAVAULT'))
    mlflow.create_experiment('Test6', 's3://visit_1')
    mlflow.set_experiment('Test6')
    mlflow.xgboost.autolog()
    with mlflow.start_run(run_name=f'XGBoost'):
        cr = CustomerRetentionTrain()
        cr.start()
