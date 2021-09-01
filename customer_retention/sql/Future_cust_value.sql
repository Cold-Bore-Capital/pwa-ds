truncate bi.future_cust_value;
drop table if exists bi.future_cust_value;
create table bi.future_cust_value(
ID      		                bigint     		identity
, uid                           varchar(10)     primary key
, final_category                 smallint
, final_category_prob            decimal(4,3)
, predicted_val_prob_0          decimal(4,3)
, predicted_val_prob_1          decimal(4,3)
, predicted_val_prob_2          decimal(4,3)
, predicted_val_prob_3          decimal(4,3)
, predicted_val_prob_4          decimal(4,3)
, predicted_val_prob_5          decimal(4,3)
, predict_for_visit_number      smallint
, total_num_visit               smallint
, total_future_spend            decimal(6,0)
, most_recent_visit_spend       decimal(6,0)
, total_past_spend              decimal(6,0)
, date_of_upload                date
);

GRANT select, update, insert, delete
    ON ALL TABLES IN SCHEMA bi
    TO pipeline;

GRANT select
    ON ALL TABLES IN SCHEMA bi
    TO looker;