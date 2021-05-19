
import pandas as pd
import numpy as np

from  cbcdb import DBManager
from dotenv import load_dotenv, find_dotenv
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
import statsmodels.api as sm
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

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
            where p.name not like '%Subscri%'
              and p.product_group != 'Surgical Services'
            group by 1, 2, 4, 5, 6, 7, 8, 9, 10;"""
        self.df = self.db.get_sql_dataframe(sql)

    def process(self, df):
        df = df.groupby(['animal_id', 'species', 'ani_age', 'datetime', 'name',
                         'type', 'tracking_level', 'product_group', 'visit_more_than_once', 'rank_'], as_index=False)['revenue'].sum()

        # get dummies for categorial features
        df_product_group = pd.get_dummies(df.product_group)
        df_species = pd.get_dummies(df.species)
        # df_ = pd.concat([df1,df_product,df_species, df_product_group]).fillna(0).drop(columns = ['species','name','product_group'])
        df_ = pd.concat([df, df_species, df_product_group], axis=1).drop(columns=['species', 'name', 'product_group'])
        df_final = df_.groupby(['animal_id',
                                'ani_age',
                                'rank_',
                                'visit_more_than_once'], as_index=False)['Canine (Dog)',
                                                                 'Feline (Cat)', 'Dentistry & Oral Surgery', 'Diagnostics',
                                                                 'Laboratory - Antech', 'Laboratory - In-house', 'Medications - Oral',
                                                                 'Parasite Control', 'Professional Services', 'Promotions', 'Vaccinations',
                                                                'Wellness Plan Fees'].sum()
        df_ = df_final[df_final.rank_==1].copy()
        return df_

    def split_train_test(self):
        X = self.df[['Canine (Dog)', #'ani_age',
               'Feline (Cat)', 'Dentistry & Oral Surgery', 'Diagnostics',
               'Laboratory - Antech', 'Laboratory - In-house', 'Medications - Oral',
               'Parasite Control', 'Professional Services', 'Promotions',
               'Vaccinations', 'Wellness Plan Fees']]
        y = self.df['return_cust'].apply(lambda x: int(x))
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X,
                                                                                y,
                                                                                test_size=.2,
                                                                                random_state=42)
    def glm_fit(self):
        res = sm.GLM(y_train, X_train, family=sm.families.Binomial()).fit()


if __name__=="__main__":
    load_dotenv(find_dotenv())
    cust_ret = CustomerRetention()
    cust_ret.read_in_latest()
    cust_ret.process()
    cust_ret.split_train_test()

