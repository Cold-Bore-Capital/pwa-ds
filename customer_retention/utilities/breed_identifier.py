import pandas as pd
import requests
import os
from cbcdb import DBManager
from dotenv import load_dotenv, find_dotenv
import numpy as np
from typing import List

load_dotenv(find_dotenv())


class BreedIdentifier():
    def __init__(self,
                 export: bool = False):
        self.db = DBManager()
        self.azure_key = os.environ.get('AZURE_KEY')

    def start(self, new_breeds: List):
        db = DBManager()

        # Remove none from list
        new_breeds = [i for i in new_breeds if i]

        # Identify tier using azure
        df = self.identify_tier(new_breeds)

        #breed groupings
        self.identify_breed_grouping(df)

        self.write_to_db(df, db)

    def identify_tier(self, new_breeds: List):
        tier = []
        tier_score = []
        for i in new_breeds:
            t, t_s = self.breed_api(i, self.azure_key)
            tier.append(t)
            tier_score.append(t_s)
        df_breeds = pd.DataFrame({'breed': new_breeds, 'tier': tier, 'tier_score': tier_score})

        return df_breeds

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
        df['breed_group'] = df.apply(lambda x: 'working' if any(ext in x['breed'].lower() for ext in working) else x['breed_group'], axis=1)
        df['breed_group'] = df.apply(
            lambda x: 'herding' if any(ext in x['breed'].lower() for ext in herding) else x['breed_group'], axis=1)
        df['breed_group'] = df.apply(
            lambda x: 'hound' if any(ext in x['breed'].lower() for ext in hound) else x['breed_group'], axis=1)
        df['breed_group'] = df.apply(
            lambda x: 'sporting' if any(ext in x['breed'].lower() for ext in sporting) else x['breed_group'], axis=1)
        df['breed_group'] = df.apply(
            lambda x: 'non_sporting' if any(ext in x['breed'].lower() for ext in non_sporting) else x['breed_group'],
            axis=1)
        df['breed_group'] = df.apply(
            lambda x: 'toy' if any(ext in x['breed'].lower() for ext in toy) else x['breed_group'], axis=1)
        df['breed_group'] = df.apply(
            lambda x: 'companion' if any(ext in x['breed'].lower() for ext in companion) else x['breed_group'], axis=1)
        df['breed_group'] = df.apply(
            lambda x: 'terrier' if any(ext in x['breed'].lower() for ext in terrier) else x['breed_group'], axis=1)
        df['breed_group'] = df.apply(
            lambda x: 'mix' if any(ext in x['breed'].lower() for ext in mix) else x['breed_group'], axis=1)
        df['breed_group'] = df.apply(
            lambda x: 'cat' if any(ext in x['breed'].lower() for ext in cat) else x['breed_group'], axis=1)

    @staticmethod
    def breed_api(animal, key):
        response = requests.get(
            f"https://pwa-breed-identification.cognitiveservices.azure.com/luis/prediction/v3.0/apps/bd914c7a-a58f-499c-a930-5d68ebbfaa95/slots/production/predict?subscription-key={key}&verbose=true&show-all-intents=true&log=true&query={animal}")
        top_intent = response.json()['prediction']['topIntent']
        score = response.json()['prediction']['intents'][top_intent]['score']
        return top_intent, score

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
