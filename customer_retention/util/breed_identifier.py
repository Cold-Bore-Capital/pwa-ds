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
        new_breeds_cleaned = [i for i in new_breeds if i]
        df = pd.DataFrame({'breed': new_breeds_cleaned})

        # Identify tier using azure
        self.identify_tier(df)

        # breed groupings
        self.identify_breed_grouping(df)

        self.write_to_db(df, db)

    def identify_tier(self, df: pd.DataFrame):
        df['tier'] = 'oth'
        df['tier_score'] = 0
        for i in range(len(df)):
            df.loc[i, ['tier', 'tier_score']] = self.breed_api(df.loc[i, 'breed'], self.azure_key)

    @staticmethod
    def identify_breed_grouping(df: pd.DataFrame):
        large = ['Greyhound', 'Bernese Mountain', 'German Shepherd', 'English Foxhound', 'Otterhound', 'Komondor',
                 'Curly Coated Retriever', 'Ibizan', 'Kuvasz', 'Briard', 'Irish Red and White', 'Beauceron',
                 'Scottish Deerhound', 'Bluetick Coonhound', 'Black Russian Terrier', 'Redbone Coonhound',
                 'Spinone Italiano', 'Tibetan Mastiff', 'Belgian Sheepdog', 'Pointer', 'Anatolian Shepherd', 'Mastiff',
                 'Black and Tan Coonhound', 'Belgian Tervuren', 'Leonberger', 'Borzoi', 'Giant', 'Gordon Setter', 'Afghan Hound', 'English Setter',
                 'Bouvier des Flandres', 'Greater Swiss Mountain Dog', 'Irish Wolfhound German', 'Wirehaired Pointer',
                 'Belgian Malinois', 'Great Pyrenees', 'Irish Setter', 'Cane Corso', 'Alaskan Malamute',
                 'Saint Bernard','Bloodhound', 'Akita', 'Chesapeake Bay Retriever', 'Rhodesian Ridgeback', 'Newfoundland',
                 'Bullmastiff','Collie', 'American English Coonhound', 'Weimaraner', 'Mastiff', 'Great Dane',
                 'German Shorthaired Pointer',
                 'Doberman Pinscher', 'Rottweiler', 'mountain', 'doberman', 'husky', 'german sheperd', 'Vizsla',
                 'bernard', 'lab', 'golden', 'large', 'primo carnera', 'retriever mix', 'blackmouth cur',
                 'black mouth cur','Dogue De Bordeaux','Dutch Shepard','rodesian Ridgeback']

        medium = ['Pug', 'Boston Terrier', 'French Bulldog', 'Poodle', 'Boxer', 'Bulldog', 'Golden Retriever',
                  'Labrador Retriever', 'English Springer Spaniel', 'American Foxhound', 'Harrier', 'Canaan Dog',
                  'Norwegian Buhund',
                  'Polish Lowland Sheepdog', 'Puli', 'Xoloitzcuintli', 'Pharaoh Hound', 'Entlebucher Mountain Dog',
                  'Field Spaniel',
                  'Clumber Spaniel', 'Kerry Blue Terrier', 'Finnish Spitz', 'Pyrenean Shepherd',
                  'American Water Spaniel', 'Irish Water Spaniel', 'Plott',
                  'Icelandic Sheepdog', 'Boykin Spaniel', 'German Pinscher', 'Irish Terrier', 'Welsh Springer Spaniel',
                  'Saluki',
                  'Bearded Collie Nova Scotia Duck Tolling Retriever', 'Finnish Lapphund', 'Keeshond',
                  'Norwegian Elkhound',
                  'Basenji', 'Wirehaired Pointing Griffon', 'Standard Schnauzer', 'Flat - Coated Retriever',
                  'Old English Sheepdog', 'Staffordshire Bull Terrier', 'Dalmatian', 'American Staffordshire Terrier',
                  'Samoyed',
                  'Chow Chow', 'English Cocker Spaniel', 'Australian Cattle', 'Whippet', 'Portuguese Water', 'Airedale',
                  'Soft - Coated Wheaten', 'Bull Terrier', 'Chinese Shar Pei', 'Border Collie', 'Vizsla Brittany',
                  'Australian Shepherd', 'Siberian Husky', 'pit', 'jack russ', 'aussie', 'bull', 'wolf', 'schnauzer',
                  'medium', 'coonhound', 'american terrier', 'belgian shepherd', 'belgian Shepard', 'auusie',
                  'Wheaten Terrier', 'tierrier mix', 'mutt', 'klee kai', 'blue heeler', 'catahoula',
                  'australian Stumpy Tail Cattle', 'carolina dog', 'Scnoodle','lagotto','Red Bone Coon','sheltie','brittany',
                  'Tennessee Hound', 'staff', 'shar pei', 'Australian Sheepdog','Blue Tick Hound']

        small = ['Chihuahua', 'Pomeranian', 'Maltese', 'Cavalier King Charles Spaniel', 'Silky Terrier',
                 'Chinese Crested Dog', 'Miniature Schnauzer', 'Bichon Frise', 'Yorkshire Terrier', 'Pekingese',
                 'Shih Tzu', 'Japanese Chin', 'Havanese', 'Miniature Pinscher', 'Brussels Griffon', 'Papillon',
                 'Affenpinscher', 'Pug', 'doodle', 'yorkie', 'shih', 'shitzu', 'poo', 'schnoodle', 'mini aussie',
                 'crested', 'pom', 'shorkie', 'mini', 'teddy', 'small', 'chorkie', 'chi', 'tibetan', 'dachs', 'toy',
                 'bichon', 'yorki', 'corgi', 'Pekingese', 'Basset Hound', 'Bichon Frise', 'West Highland White Terrier',
                 'Havanese', 'Cocker Spaniel', 'Pembroke Welsh Corgi', 'Maltese', 'Cavalier King Charles Spaniel',
                 'Pomeranian', 'Chihuahua', 'Shih Tzu', 'Dachshund', 'Yorkshire Terrier', 'Norwegian Lundehund',
                 'Sealyham Terrier', 'Sussex Spaniel', 'Glen of Imaal Terrier', 'Swedish Vallhund',
                 'Parson Russell Terrier', 'Skye Terrier', 'Dandie Dinmont Terrier', 'Affenpinscher', 'Lakeland Terrier', 'Bedlington Terrier',
                 'Petit Basset', 'Griffon Vendeen', 'English Toy Spaniel', 'Miniature Bull Terrier',
                 'Australian Terrier', 'Norfolk Terrier', 'Manchester Terrier', 'American Eskimo Dog', 'Tibetan Spaniel',
                 'Smooth Fox Terrier', 'Cesky Terrier', 'Schipperke', 'Toy Fox Terrier', 'Wire Fox Terrier', 'Welsh Terrier',
                 'Norwich Terrier','Tibetan Terrier', 'Silky Terrier', 'Border Terrier', 'Japanese Chin', 'Brussels Griffon',
                 'Italian Greyhound','Lhasa Apso', 'Chinese Crested', 'Cairn Terrier', 'Scottish Terrier', 'Shiba Inu',
                 'Miniature Pinscher','doxen','Papillon', 'Shetland Sheepdog', 'Miniature Schnauzer', 'Beagle', 'Shi Zu','min pin',
                 'king charles', 'rat', 'Bruxellois', 'morkie','chug']
        other_animal = ['bird', 'cotton tail bunny']
        cat = ['cat', 'siamese', 'domestic short', 'tortie', 'dsh', 'dhs', 'american shorthair', 'american longhair','sem-longhair',
               'Norwegian Forest', 'tabby', 'American Long Hair','Asian Semi-Longhair']

        small = [x.lower() for x in small]
        medium = [x.lower() for x in medium]
        large = [x.lower() for x in large]
        other_animal = [x.lower() for x in other_animal]
        cat = [x.lower() for x in cat]

        df['breed_size'] = 'oth'
        df['breed_size'] = df.apply(
            lambda x: 'small' if any(ext in x['breed'].lower() for ext in small) else x['breed_size'], axis=1)
        df['breed_size'] = df.apply(
            lambda x: 'medium' if any(ext in x['breed'].lower() for ext in medium) else x['breed_size'], axis=1)
        df['breed_size'] = df.apply(
            lambda x: 'large' if any(ext in x['breed'].lower() for ext in large) else x['breed_size'], axis=1)
        df['breed_size'] = df.apply(
            lambda x: 'cat' if any(ext in x['breed'].lower() for ext in cat) else x['breed_size'], axis=1)
        df['breed_size'] = df.apply(
            lambda x: 'other_animal' if any(ext in x['breed'].lower() for ext in other_animal) else x['breed_size'],
            axis=1)

        # df[df.breed_size == 'oth'].breed.nunique()
        # df[df.breed_size == 'oth'].breed.unique()

        # all else get's thrown into medium
        df['breed_size'] = df.apply(lambda x: 'medium' if x['breed_size'] == 'oth' else x['breed_size'], axis=1)


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
