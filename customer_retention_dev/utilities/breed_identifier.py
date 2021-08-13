import pandas as pd
import requests
import os
from cbcdb import DBManager
from dotenv import load_dotenv, find_dotenv
import numpy as np

load_dotenv(find_dotenv())


class BreedIdentifier():
    def __init__(self,
                 export: bool = False):
        self.db = DBManager()
        self.azure_key = os.environ.get('AZURE_KEY')

    def start(self):
        db = DBManager()
        df_breed = db.get_sql_dataframe("select * from bi.breeds")

        # Read in latest new species
        df = bi.read_in_latest()

        # Identify tier using azure
        df = bi.identify_tier(df, df_breed)

        bi.identify_breed_grouping(df)

        bi.write_to_db(df, db)

    def read_in_latest(self):
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
                                        
            select 
                distinct 
                 f1.breed
            from (
                         select f.uid
                          , f.breed
                          , cd.visit_number
                     from (
                              select t.location_id || '_' || t.animal_id                                                         as uid
                                   , a.breed
                                   , max(
                                      date_diff('years', timestamp 'epoch' + a.date_of_birth * interval '1 second',
                                                current_date))                                                                   as ani_age
                                   --, min(trunc(t.datetime_date)) over (partition by t.location_id || '_' || t.animal_id) as      date_of_first_visit
                                   , case when trunc(t.datetime_date) - min(trunc(t.datetime_date)) over (partition by t.location_id || '_' || t.animal_id) > 548 then 0
                                       else 1 end as less_than_1_5_yeras
                                   , trunc(t.datetime_date)                                                                      as date
                                   , a.weight
                                   , p.is_medical
                                   , p.product_group
                                   , case
                                         when apt.type_id like 'Grooming%' then 'groom'
                                         when apt.type_id like '%Neuter%' then 'neurtering'
                                         when apt.type_id like '%ental%' then 'dental'
                                         else apt.type_id end                                                                    as type_id
                                   --, p.name
                                   --, p.type
                                   --, p.tracking_level
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
                                   --where p.name not like '%Subscri%'
                                   --  and p.product_group != 'Surgical Services'
                                   --and a.breed != '0.0'
                              group by 1, 2, 5, 6, 7, 8, 9, 10) f
                      left join consecutive_days cd
                                on f.uid = cd.uid
                                    and f.date = cd.datetime_
                      left join wellness w
                                on f.uid = w.uid
                                    and f.date = w.datetime_
                        where less_than_1_5_yeras = 1) f1
            where f1.visit_number = 1"""
        df = self.db.get_sql_dataframe(sql)
        return df

    def identify_tier(self, df: pd.DataFrame, df_breed: pd.DataFrame):
        # First run
        if len(df_breed) == 0:
            list_of_breeds = df.breed.values
            tier = []
            for i in list_of_breeds:
                tier.append(self.breed_api(i, self.azure_key))
            df_breeds = pd.DataFrame({'breed': list_of_breeds, 'tier': tier})
            return df_breeds

        # All other runs
        else:
            list_of_breeds = df.breed.values
            list_of_breeds_orig = df_breed.breed.values
            new_breeds = list_of_breeds_orig - list_of_breeds

            tier = []
            for i in new_breeds:
                tier.append(self.breed_api(i, self.azure_key))

            df_breeds = pd.DataFrame({'breed': new_breeds,
                                      'tier': tier})
            df_breeds_orig = pd.concat([df_breed, df_breeds], ignore_index=True)
            return df_breeds_orig

    @staticmethod
    def identify_breed_grouping(df: pd.DataFrame):
        working = ['Alaskan Malamute', 'Siberian Huskie', 'husky', 'Great Dane', 'Doberman', 'Rottweiler', 'Akita',
                   'Anatolian Shepherd', 'Huskie', 'Saint Bernard', 'Mastiff', 'Bernard', 'Portuguese Water Dog',
                   'German Pinscher', 'Great Pyrenee', 'Giant Schnauzer', 'Greater Swiss Mountain Dog',
                   'Newfoundland', 'Samoyed', 'Bullmastiff', 'Bernese Mountain Dog', 'mountain curr', 'point',
                   'Large', 'German Shepherd', 'Belgian Malinoi']
        herding = ['Australian Cattle Dog', 'Australian Shepherd', 'Collie', 'Shetland Sheepdog',
                   'Pembroke Welsh Corgi', 'Cardigan Welsh Corgi', 'Old English Sheepdog', 'Belgian Tervuren',
                   'Canaan Dog', 'Briard', 'Bouvier des Flandre', 'corgi',
                   'boxer', 'sheep', 'shep', 'aussie', 'shetland', 'auusie']
        hound = ['Basset Hound', 'Saluki', 'Beagle', 'Harrier', 'American Foxhound', 'English Foxhound',
                 'Bloodhound', 'Irish Wolfhound', 'Dachshund', 'Otterhound', 'Norwegian Elkhound', 'Greyhound',
                 'Italian Greyhound', 'Whippet', 'Afghan Hound', 'Borzois Hound', 'Coonhound',
                 'Rhodesian Ridgeback', 'Petit Basset Griffon Vendéen', 'Basenji', 'hound']
        sporting = ['Cocker', 'Irish Setter', 'English Springer Spaniel', 'Clumber Spaniel',
                    'German Shorthaired Pointer', 'German Wirehaired Pointer', 'American Water Spaniel',
                    'Weimaraner', 'Retriever', 'Chesapeake Bay Retriever', 'English Setter', 'staffordshire']
        non_sporting = ['Dalmatian', 'Chow Chow', 'Finnish Spitz', 'Shar Pei', 'American Bulldog', 'Poodle',
                        'Boston Terrier', 'Lhasa Apso', 'Shiba Inu', 'French Bulldog', 'Schipperke',
                        'American Eskimo Dog']
        toy = ['Chihuahua', 'Pomeranian', 'Maltese', 'Cavalier King Charles Spaniel', 'Silky Terrier',
               'Chinese Crested Dog', 'Miniature Schnauzer', 'Bichon Frise', 'Yorkshire Terrier', 'Pekingese',
               'Shih Tzu', 'Japanese Chin', 'Havanese', 'Miniature Pinscher', 'Brussels Griffon', 'Papillon',
               'Affenpinscher', 'Pug', 'doodle', 'yorkie', 'shih', 'shitzu', 'poo', 'schnoodle', 'mini aussie',
               'crested', 'pom', 'shorkie', 'mini', 'teddy', 'small', 'chorkie', 'chi', 'tibetan', 'dachs', 'toy',
               'bichon', 'yorki']
        terrier = ['Airedale Terrier', 'American Staffordshire Terrier', 'Jack Russell', 'Bull Terrier', 'pit bull',
                   'pitbull' 'Fox Terrier', 'Wheaten Terrier', 'Cairn Terrier', 'West Highland White Terrier',
                   'Australian Terrier', 'Border Terrier', 'Staffordshire Bull Terrier', 'Bedlington Terrier',
                   'Kerry Blue Terrier', 'Rat Terrier', 'Scottish Terrier', 'Bull', 'blue terr', "terrier", 'pit']
        companion = ['mutt', 'goldon', 'Lab']
        mix = ['mix']
        cat = ['cat', 'siamese']

        working = [x.lower() for x in working]
        herding = [x.lower() for x in herding]
        hound = [x.lower() for x in hound]
        sporting = [x.lower() for x in sporting]
        non_sporting = [x.lower() for x in non_sporting]
        toy = [x.lower() for x in toy]
        companion = [x.lower() for x in companion]
        terrier = [x.lower() for x in terrier]

        df['breed_group'] = 'oth'
        df['breed_group'] = df.apply(lambda x: 'working' if any(ext in x['breed'].lower() for ext in working) else x['breed_group'], axis = 1)
        df['breed_group'] = df.apply(lambda x: 'herding' if any(ext in x['breed'].lower() for ext in herding) else x['breed_group'], axis=1)
        df['breed_group'] = df.apply(lambda x: 'hound' if any(ext in x['breed'].lower() for ext in hound) else x['breed_group'], axis=1)
        df['breed_group'] = df.apply(lambda x: 'sporting' if any(ext in x['breed'].lower() for ext in sporting) else x['breed_group'], axis=1)
        df['breed_group'] = df.apply(lambda x: 'non_sporting' if any(ext in x['breed'].lower() for ext in non_sporting) else x['breed_group'], axis=1)
        df['breed_group'] = df.apply(lambda x: 'toy' if any(ext in x['breed'].lower() for ext in toy) else x['breed_group'], axis=1)
        df['breed_group'] = df.apply(lambda x: 'companion' if any(ext in x['breed'].lower() for ext in companion) else x['breed_group'], axis=1)
        df['breed_group'] = df.apply(lambda x: 'terrier' if any(ext in x['breed'].lower() for ext in terrier) else x['breed_group'], axis=1)
        df['breed_group'] = df.apply(lambda x: 'mix' if any(ext in x['breed'].lower() for ext in mix) else x['breed_group'], axis=1)
        df['breed_group'] = df.apply(lambda x: 'cat' if any(ext in x['breed'].lower() for ext in cat) else x['breed_group'], axis=1)

    @staticmethod
    def breed_api(animal, key):
        response = requests.get(
            f"https://pwa-breed-identification.cognitiveservices.azure.com/luis/prediction/v3.0/apps/e57c5e2f-a645-42cb-9777-f7f2ca6ad945/slots/production/predict?subscription-key={key}&verbose=false&show-all-intents=false&log=true&query={animal}")
        return response.json()['prediction']['topIntent']

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
            sql, params = db.build_sql_from_dataframe(df, 'breeds', schema)
            db.insert_many(sql, params)


if __name__ == '__main__':
    bi = BreedIdentifier()
    bi.start()
