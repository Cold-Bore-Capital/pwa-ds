truncate bi.future_cust_value;
drop table if exists bi.future_cust_value;
create table bi.future_cust_value(
ID      		          bigint     		identity
, uid                     varchar(10)
, final_category           smallint
, final_category_prob      decimal(4,3)
, predicted_val_prob_0    decimal(4,3)
, predicted_val_prob_1    decimal(4,3)
, predicted_val_prob_2    decimal(4,3)
, predicted_val_prob_3    decimal(4,3)
, predicted_val_prob_4    decimal(4,3)
)