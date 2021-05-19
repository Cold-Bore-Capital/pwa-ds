-- Original
select t.animal_id
      , trunc(t.datetime_date)                   as date
      , sum(t.revenue)                           as revenues
      , dense_rank() over (partition by t.animal_id order by date asc) as counts
 from bi.transactions t
          left join bi.divisions d
                    on d.division_id = t.division_id
          inner join bi.animals a
                     on a.id = t.animal_id
     -- 106 records with the join below
     --and a.location_id = t.location_id
          inner join bi.products p
                     on t.product_id = p.ezyvet_id
                         and t.location_id = p.location_id
                         and p.is_medical = 1
 where is_dead = 0
   and a.active = 1
 group by 1, 2
 having revenues > 100;



select t.animal_id
     , a.species
     , trunc(t.datetime_date) as date
     , sum(t.revenue)         as revenues
     , count(1)               as counts
from bi.transactions t
         left join bi.divisions d
                   on d.division_id = t.division_id
         inner join bi.animals a
                    on a.id = t.animal_id
         inner join bi.products p
                    on t.product_id = p.ezyvet_id
                        and t.location_id = p.location_id
                        and p.is_medical = 1
where is_dead = 0
  and a.active = 1
group by 1, 2, 3
having counts >= 2
   and revenues > 100
order by counts desc;



DROP TABLE IF EXISTS test;
create temporary table test as (
    select t.animal_id
         , a.species
         , max(date_diff('years', timestamp 'epoch' + a.date_of_birth * interval '1 second', current_date)) over (partition by t.animal_id) as ani_age
         , trunc(t.datetime_date)                                                                           as datetime_
         , a.weight
         , p.name
         , p.type
         , p.tracking_level
         , p.product_group
         -- Identify animals that have at least 2 visits where revenue was higher than 100 both times and
         -- product type doesn't contain surgical services or subscriptions
         , case
               when a.id in (select distinct x.animal_id
                             from (
                                      select t.animal_id
                                           , trunc(t.datetime_date)                   as date_
                                           , sum(t.revenue)                           as revenues
                                           , dense_rank() over (partition by t.animal_id order by date_ asc) as num_of_visits
                                      from bi.transactions t
                                               left join bi.divisions d
                                                         on d.division_id = t.division_id
                                               inner join bi.animals a
                                                          on a.id = t.animal_id
                                                            and a.location_id = t.location_id
                                               inner join bi.products p
                                                          on t.product_id = p.ezyvet_id
                                                              and t.location_id = p.location_id
                                                              and p.is_medical = 1
                                      where is_dead = 0
                                        and a.active = 1
                                        and p.name not like '%Subscri%'
                                        and p.product_group != 'Surgical Services'
                                      group by 1, 2
                                      having revenues > 100) x
                             where x.num_of_visits > 1)
                   then 1
               else 0 end                                                                                   as visit_more_than_once
         --, dense_rank() over (partition by t.animal_id order by datetime_ asc)                               as rank_
         , sum(t.revenue) over (partition by t.animal_id, trunc(t.datetime_date) )                            as rev_partition
         --, sum(t.revenue)                                                                                   as revenue
    from bi.transactions t
             inner join bi.products p
                        on t.product_id = p.ezyvet_id
                            and t.location_id = p.location_id
                            and p.is_medical = 1
             inner join bi.animals a
                        on a.id = t.animal_id
                        and a.location_id = t.location_id
             left join bi.contacts c
                       on a.contact_id = c.ezyvet_id
                           and t.location_id = c.location_id
    where p.name not like '%Subscri%'
      and p.product_group != 'Surgical Services');


select
    animal_id
    , species
    , ani_age
    , datetime_
    , weight
    , name
    , type
    , tracking_level
    , product_group
    , visit_more_than_once
    , dense_rank() over (partition by animal_id order by datetime_) as num_of_visits
    , rank_rev
from test
where rank_rev > 100;




select
    f.animal_id
    , f.species
    , f.ani_age
    , f.datetime
    , f.weight
    , f.name
    , f.type
    , f.tracking_level
    , f.product_group
    , f.visit_more_than_once
    , f.rev_partition
    , sum(f.revenue)                                                as revenue
    from (
            select
                t.animal_id
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
                        then 1 else 0 end                                                                       as visit_more_than_once
                    , t.revenue                                                                                 as revenue
                    , sum(t.revenue) over (partition by t.animal_id, trunc(t.datetime_date) )                   as rev_partition

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
                group by 1, 2, 4, 5, 6, 7, 8, 9, 10, 11) f
    where f.rev_partition > 100
    group by 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11