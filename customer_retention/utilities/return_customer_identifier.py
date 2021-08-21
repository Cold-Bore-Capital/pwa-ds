import pandas as pd
import requests
import os
from cbcdb import DBManager
from dotenv import load_dotenv, find_dotenv
import numpy as np
from typing import List

load_dotenv(find_dotenv())


class ReturnCustomerIdentifier():
    def __init__(self):
        self.db = DBManager()

    def start(self):
        db = DBManager()

        # Sql to see if customers return
        df = self.read_in_latest(db)

        # Identify customers to update
        df = self.identify_return_customers_to_update(df, db)

        # Identify tier using azure
        db.update_batch_from_df(df,
                                update_cols=['total_future_spend','total_num_visit'],
                                static_cols=['id'],
                                schema='bi',
                                table='future_cust_value')


    def read_in_latest(self, db: DBManager):
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
                                    
     select f.uid
      , f.breed
      , f.ani_age
      , f.date
      , f.weight
      , f.is_medical
      , f.product_group
      , f.product_name
      , cd.visit_number
      , cd.visit_more_than_once
      , cd.max_num_visit as total_num_visit
      , f.revenue
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
              group by 1, 2, 6, 7, 8, 9, 10) f
            inner join consecutive_days cd
                on f.uid = cd.uid
                and f.date = cd.datetime_
            inner join bi.future_cust_value fcv 
                on f.uid = fcv.uid
            where cd.visit_number > 1  
            """
        df = db.get_sql_dataframe(sql)
        return df

    @staticmethod
    def identify_return_customers_to_update(df: pd.DataFrame, db: DBManager):
        """
        Args:
            df: Cleaned dataframe to join with bi.cust_details to identify new customers.
            db: DBManager connected to redshift
        Returns:
            None

        """
        # Filter df
        df_ = df[['total_num_visit','total_future_spend','uid']].drop_duplicates()

        # Check to see if there are new customers
        cust_db = db.get_sql_dataframe("Select * from bi.future_cust_value")[
            ['id', 'uid', ]]  # , 'cust_loc_num'

        df_ = df_.merge(cust_db,
                       on=['uid'],
                       how='left',
                       indicator=True)
        df_ = df_[df_._merge=='both'].drop(columns=['_merge','uid'])
        return df_


if __name__ == '__main__':
    rci = ReturnCustomerIdentifier()
    rci.start()
