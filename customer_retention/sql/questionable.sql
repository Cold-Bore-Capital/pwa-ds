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
                  , rank() over (partition by uid order by rank_group asc) as visit_number
             from (
                      SELECT uid
                           , datetime                                                                  as datetime_
                           , dateadd(day, -rank() OVER (partition by uid ORDER BY datetime), datetime) AS rank_group
                      FROM (SELECT DISTINCT t.location_id || '_' || t.animal_id as uid
                                          , trunc(t.datetime_date)              as datetime
                            from bi.transactions t
                                     inner join bi.animals a
                                                on a.ezyvet_id = t.animal_id
                                                    and a.location_id = t.location_id
                      where t.revenue > 0
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


drop table questionable_behav;
create temporary table questionable_behav as (
select
f.uid
, f.date
, f.is_medical
, f.wellness_plan
, f.wellness_plan_count
, f.product_name
, f.free_count
, f.revenue
, f.rank_
from (
    select
    t.location_id || '_' || t.animal_id                                                         as uid
    , trunc(t.datetime_date)                                                                      as date
    , w.wellness_plan
    , sum(case when w.wellness_plan is not null then 1 else 0 end)
     over (partition by t.location_id || '_' || t.animal_id    )  as wellness_plan_count
    , p.is_medical
    , t.product_name
    ,  sum(case when t.product_name like 'First Day Daycare Fre%' then 1 else 0 end)
     over (partition by t.location_id || '_' || t.animal_id    )  as free_count
    , t.revenue                                                                                   as revenue
    , dense_rank()
     over (partition by t.location_id || '_' || t.animal_id order by trunc(t.datetime_date) asc) as rank_
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
          inner join consecutive_days cd
                    on t.location_id || '_' || t.animal_id = cd.uid
                        and trunc(t.datetime_date) = cd.datetime_
          left join wellness w
                    on t.location_id || '_' || t.animal_id = w.uid
                        and trunc(t.datetime_date) = w.datetime_) f
--where f.free_count > 5);
