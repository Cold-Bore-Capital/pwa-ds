truncate table bi.breeds;
drop table if exists bi.breeds;
create table bi.breeds(
ID      				bigint     		identity
, breed				    varchar(50)		NOT NULL
, tier					varchar(25)
, tier_score            decimal(5,4)
, breed_size           varchar(25)
);