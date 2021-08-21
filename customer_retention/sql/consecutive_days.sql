-- medical visit
-- pull doctor's notes
-- look into type_id
-- (look for trauma and acute) probably not coming back
-- routines, vaccines, nutures will probably come back
-- wellness plan into a binary indicator
-- look into conversions for 2nd to 3rd etc, not only 1st conversion.
drop table if exists consecutive_days;
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

select * from consecutive_days
where uid = '2_884';

select
uid
, datetime_
, visit_number
, dense_rank() over (order by visit_number asc)
from consecutive_days
    where uid = '2_884';



drop table if exists wellness;
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
drop table if exists x;
create temporary table x as (
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
                                   left join bi.animals a
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
        order by 1, 4);
--where product_name like  'First Day Daycare Free%';

select
visit_number
 from x;



























drop table consecutive_days;
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
                  , rank() over (partition by uid order by rank_group asc) as visit_number
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

drop table if exists wellness;
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
        left join bi.future_cust_value fcv
            on f1.uid = fcv.uid
        where f1.visit_number = 1
            and fcv.uid is null
        order by 1, 4;