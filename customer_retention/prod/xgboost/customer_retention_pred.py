import pandas as pd
import numpy as np

from cbcdb import DBManager
from datetime import date
from dotenv import find_dotenv, load_dotenv
import mlflow
import os
import xgboost as xgb
import mlflow.xgboost
import sys
sys.path.append(os.getcwd())
from customer_retention.util.breed_identifier import BreedIdentifier

load_dotenv(find_dotenv())


class CustomerRetentionPred():
    def __init__(self):
        self.azure_key = os.environ.get('AZURE_KEY')

    def start(self,
              visit_number,
              objective='multi:softprob'):

        db = DBManager()
        # Read in the latest data
        df = self.read_in_latest(db, visit_number)

        if isinstance(df, str):
            return print(df)

        # Find mean weight of animals to inject if weight is NA
        df = self.fill_in_weight(df, db)

        # Return dog Breed
        df = self.dog_breed(df, db)

        # Begin feature engineering
        df = self.feature_eng(df, objective)

        # Pull Best model
        df = self.pull_best_model_and_predict(df, visit_number)

        # Write to db
        self.write_to_db(df, db)

    def read_in_latest(self, db: DBManager, visit_number: int):
        sql = f"""    
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
             , f1.total_future_spend
             , f1.most_recent_visit_spend
             , f1.total_past_spend
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
                       , sum(case when cd.visit_number > {visit_number} then f.revenue else 0 end)
                        over (partition by f.uid) as total_future_spend
                      , sum(case when cd.visit_number = {visit_number} then f.revenue else 0 end)
                        over (partition by f.uid) as most_recent_visit_spend
                        , sum(case when cd.visit_number < {visit_number} then f.revenue else 0 end)
                        over (partition by f.uid) as total_past_spend
                 from (

                          select t.location_id || '_' || t.animal_id                                                         as uid
                               , a.breed
                               , max(
                                  date_diff('years', timestamp 'epoch' + a.date_of_birth * interval '1 second',
                                            current_date))                                                                   as ani_age
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
                          group by 1, 2, 4, 5, 6, 7, 8
                     ) f
                  inner join consecutive_days cd
                            on f.uid = cd.uid
                                and f.date = cd.datetime_
                  left join wellness w
                            on f.uid = w.uid
                                and f.date = w.datetime_) f1
        left join bi.future_cust_value fcv 
            on f1.uid = fcv.uid
            and fcv.predict_for_visit_number = {visit_number}
        where f1.visit_number = {visit_number}
            and fcv.uid is null  
        order by 1, 4;
        """
        df = db.get_sql_dataframe(sql)
        if len(df) == 0:
            return f"No new patients to be processed with visit number {visit_number}"
        return df

    @staticmethod
    def fill_in_weight(df: pd.DataFrame, db: DBManager) -> pd.DataFrame:
        sql = """select x.breed
                         , x.baby
                         , avg(weight) as weight
                    from (
                             select breed
                                  , date((convert_timezone('utc', 'america/chicago',
                                                           timestamp 'epoch' +
                                                           (a.date_of_birth) * interval '1 second'))) as dob
                                  , datediff(year, dob, current_date)                                    animal_age
                                  , case
                                        when animal_age <= 1 then 'baby'
                                        else 'adult'
                                 end                                                                  as baby
                                  , weight
                             from bi.animals a
                                      join bi.transactions t
                                           on a.id = t.animal_id) x
                    group by 1, 2;"""
        df_weight = db.get_sql_dataframe(sql)

        df['weight'] = df['weight'].fillna(0)
        df_w_weight = df[df.weight > 0]
        df_wo_weight = df[df.weight == 0]
        df_wo_weight.drop(columns='weight',inplace=True)
        df_wo_weight['baby'] = df_wo_weight['ani_age'].apply(lambda x: 'baby' if x < 1 else 'adult')

        df_wo_weight = df_wo_weight.merge(df_weight, on=['breed','baby'])
        df_wo_weight.drop(columns=['baby'], inplace=True)

        df_ = pd.concat([df_w_weight, df_wo_weight], ignore_index=True)
        return df_

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
    def feature_eng(df: pd.DataFrame, df_weight: pd.DataFrame) -> pd.DataFrame:
        df.reset_index(drop=True, inplace=True)
        df_ = df[['uid', 'ani_age', 'weight', 'product_group', 'breed_size', 'tier', 'is_medical',
                   'wellness_plan', 'total_future_spend','most_recent_visit_spend', 'total_past_spend','max_num_visit']]

        # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
        # # Create categorical df. Number of rows should equate to the number of unique uid
        # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

        cat_features = ['wellness_plan', 'breed_size', 'tier']#, 'product_group']
        df_cat = df_[['uid'] + cat_features].drop_duplicates()
        # Come back to use this when we have more data
        #df_product_group = pd.get_dummies(df_.product_group)
        df_breed_size = pd.get_dummies(df_cat.breed_size)
        df_tier = pd.get_dummies(df_cat.tier)
        df_tier.rename(columns={'None': 'tier_other'}, inplace=True)

        df_cat = pd.concat([df_cat[['uid']],
                            # df_cat.product_group,
                            df_breed_size,
                            df_tier], axis=1)  #
        df_categories = df_cat.groupby(['uid'], as_index=False).max()

        # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
        # Create standardized df. Number of rows should equate to the number of unique uid
        # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
        cont_features = ['ani_age', 'weight', 'breed_size', 'is_medical', 'wellness_plan','most_recent_visit_spend',
                         'total_future_spend', 'total_past_spend', 'max_num_visit']
        df_cont = df_[['uid'] + cont_features].drop_duplicates()

        # look into a binary indicator for weight
        df_cont['ani_age'] = df_cont['ani_age'].fillna((df_cont['ani_age'].mean()))

        df_cont['weight'] = df_cont.groupby(['ani_age', 'breed_size'])['weight'].transform(
            lambda x: x.fillna(x.mean()))


        df_cont = df_cont.groupby(['uid', 'ani_age', 'weight', 'total_future_spend',
                                   'most_recent_visit_spend', 'total_past_spend','max_num_visit'],
                                  as_index=False).agg(
            is_medical_max=('is_medical', 'max'),
            is_medical_count=('is_medical', 'count'),
            wellness_plan_max=('wellness_plan', 'max'),
            wellness_plan_count=('wellness_plan', 'count')
        )

        df_final = df_cont.merge(df_categories, on='uid')
        return df_final

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
    def pull_best_model_and_predict(df_orig: pd.DataFrame,  visit_number: int):
        mlflow.get_experiment_by_name('visit_' + str(visits_number))
        # pull the best, most recent model
        models = mlflow.search_runs(order_by=["metrics.Precision_Binary DESC", "attribute.start_time DESC"])
        model = mlflow.xgboost.load_model(models.loc[0]['artifact_uri']+'/model')
        mlflow.xgboost.load_model(models.loc[0]['artifact_uri']+'/model')
        final_columns = list(df_orig.columns)
        for i in ['uid','total_future_spend','max_num_visit']:
            final_columns.remove(i)
        X = df_orig[final_columns].copy()

        # Predict on current test data
        test = xgb.DMatrix(X, missing=-999.0)
        y_pred = model.predict(test)

        # Format data to be put into db
        y_pred = np.around(y_pred, decimals=3)
        y_final_prob = [max(i) for i in y_pred]
        y = np.argmax(y_pred, axis=-1)
        df = pd.DataFrame(y_pred, columns=['predicted_val_prob_0', 'predicted_val_prob_1', 'predicted_val_prob_2',
                                           'predicted_val_prob_3', 'predicted_val_prob_4', 'predicted_val_prob_5'])
        df['uid'] = df_orig['uid']
        df['final_category'] = y
        df['final_category_prob'] = y_final_prob
        df['total_num_visit'] = df_orig['max_num_visit']
        df['predict_for_visit_number'] = visit_number
        df['total_past_spend'] = df_orig['total_past_spend']
        df['total_future_spend'] = df_orig['total_future_spend']
        df['date_of_upload'] = date.today().strftime('%Y-%m-%d')
        df['most_recent_visit_spend'] = df_orig.most_recent_visit_spend

        return df

    @staticmethod
    def write_to_db(df: pd.DataFrame, db: DBManager, schema: str = 'bi') -> None:
        """
        Args:
            df: Cleaned and processed dataframe to be inserted into redshift
            db: DBManager connected to redshift
            schema: Schema to be used when writing to the db
        Returns:
            None
        """
        if len(df) > 0:
            sql, params = db.build_sql_from_dataframe(df, 'future_cust_value', schema)
            db.insert_many(sql, params)


if __name__ == '__main__':
    mlflow.set_tracking_uri(os.environ.get('MLFLOW__CORE__SQL_ALCHEMY_CONN'))
    visits_number = [1, 2, 3, 4, 5]
    visits_name = ['visit_' + str(x) for x in visits_number]
    for v_num, v_name in zip(visits_number, visits_name):
        mlflow.set_experiment(v_name)
        cr = CustomerRetentionPred()
        cr.start(visit_number=v_num)
