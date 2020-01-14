import recordlinkage
import pandas as pd
import csv
import re
import pymongo
from pymongo import MongoClient

#path to our datasets
ORIGINAL = "restaurants.tsv"
DUPLICATES = "restaurants_DPL.tsv"

#parse to tsv files into a dataframe
df = pd.read_csv(ORIGINAL, sep='\t')
df_DPL = pd.read_csv(DUPLICATES, sep='\t')

#add 1 to index to it matches the id
df.index += 1
df_DPL.index += 1

#creating the gold standard from the duplicates list
to_drop = pd.Series(df_DPL['id1'])
gold = df[~df['id'].isin(to_drop)]

#function for removing characters after a certain index
def remove(name,index):
    subname = name[:index]
    return subname

''' Function for cleaning up street names '''
#ignore all cases and special characters
street_type_re = re.compile(r'\b\S+\.?$', re.IGNORECASE)

#the mapping from abbreviated street types to the full, corrected type
street_mapping = {"av": "avenue",
           "av.": "avenue",
           "ave": "avenue",
           "ave.": "avenue",
           "blv": "boulevard",
           "blv.": "boulevard",
           "blvd": "boulevard",
           "blvd.": "boulevard",
           "st": "street",
           "st.": "street",
           "rd": "road",
           "rd.": "road",
           "dr": "drive",
           "dr.": "drive",
           "pkwy": "parkway",
           "pky": "parkway"}

def clean_street(street_name):
    #if found comma, remove everything after that
    try:
        comma_index = street_name.index(',')
        street_name = remove(street_name,comma_index)
    except:
        pass

    #if found #, remove everything after that
    try:
        pound_index = street_name.index('#')
        street_name = remove(street_name, pound_index)
    except:
        pass
        
    #remove special characters
    try:
        street_name = re.sub('\.','', street_name)
    except:
        pass
    
    try:
        street_name = re.sub(' +', ' ', street_name)
    except:
        pass
        
    #mapping of direction abbreviations to full
    try:
        street_name = street_name.replace(' n ', ' north ')
        street_name = street_name.replace(' w ', ' west ')
        street_name = street_name.replace(' e ', ' east ')
        street_name = street_name.replace(' s ', ' south ')
        street_name = street_name.replace(' ne ', ' northeast ')
        street_name = street_name.replace(' nw ', ' northwest ')
        street_name = street_name.replace(' se ', ' southeast ')
        street_name = street_name.replace(' sw ', ' southwest ')
    except:
        pass
    
    #mapping of street names
    try:
        name_array = street_name.split(' ')
        last = name_array[-1]
        name_array[-1] = street_mapping[last]
        joined = ' '.join(name_array)
        return joined
    except:
        return street_name
    
    return street_name


''' Function for cleaning up city names '''
#the mapping from problem cities encountered to their corrected values
city_mapping = {'w hollywood': 'hollywood',
                'w. hollywood': 'hollywood',
                'la ': 'los angeles',
                'west la': 'los angeles',
                'new york city': 'new york'}

def clean_city(city_name):
    #mapping of city names
    for key, value in city_mapping.items():
        city_name = city_name.replace(key, value)
        
    #remove special characters
    city_name = re.sub('\.', ' ', city_name)
 
    return city_name

''' Function for cleaning up restaurant names '''
def clean_name(name): 
    #remove characters after (
    try: 
        index = name.index('(') 
        name = remove(name, index) 
    except: 
        pass
    
    #remove characters after 'on'
    try:
        index = name.index(' on ')
        name = remove(name, index)
    except:
        pass

    #substitute characters and removing descriptor words
    try:
        name = name.replace("&", "and")
        name = name.replace("grille", "grill")

        name = name.replace("'s", "")
        name = name.replace(" the", "")
        name = name.replace("the ", "")
        name = name.replace(" and ", "")
        name = name.replace("hotel", "")
        name = name.replace("restaurant", "")
        name = name.replace("bistro", "")
    except:
        pass
    
    return name

''' Function for cleaning up restaurant types '''
def clean_type(type_name):
    #remove characters after (
    try:
        index = type_name.index('(')
        type_name = remove(type_name, index)
    except:
        pass

    #remove characters after /
    try:
        index = type_name.index('/')
        type_name = remove(type_name, index)
    except:
        pass
    
    return type_name

''' Function for cleaning up phone numbers '''
def clean_phone(phone):
    #remove all special characters
    phone = re.sub('\W', '', phone)
    return phone

''' Function to clean the dataset and remove extra spaces '''
def audit(df):
    for index, row in df.iterrows():
        df.loc[index, 'address'] = clean_street(row['address']).strip()
        df.loc[index, 'city'] = clean_city(row['city']).strip()
        df.loc[index, 'phone'] = clean_phone(row['phone']).strip()
        df.loc[index, 'name'] = clean_name(row['name']).strip()
        df.loc[index, 'type'] = clean_type(row['type'])
        
audit(df)

''' Indexer to create record pairs '''
indexer = recordlinkage.Index()
indexer.block(on=['name'])
pairs = indexer.index(df)

''' Comparing the strings in the record pairs '''
compare_cl = recordlinkage.Compare()
compare_cl.string('name', 'name', method='smith_waterman', threshold=0.75, label='name')
compare_cl.string('address', 'address', method='damerau_levenshtein', threshold=0.75, label='address')
compare_cl.string('city', 'city', method='jarowinkler', threshold=0.75, label='city')
compare_cl.exact('phone', 'phone', label='phone')

#comparing the features against the original dataframe
features = compare_cl.compute(pairs, df)

#if more than 2 criteria matched, the record is a match
matches = features[features.sum(axis=1) >= 2.0]

''' Creating a dataframe from the matching pairs '''
idone = matches.unstack(level=0)
idtwo = matches.unstack(level=1)
res = pd.DataFrame(list(zip(idone.index,idtwo.index)), columns=['id1', 'id2'])

#true positives
tp = df_DPL

#intersection between two dataframes to find the false positives and the false negatives
merged = pd.merge(tp, res, how='outer', on=['id1', 'id2'], indicator=True)
fp = merged[merged['_merge'] == 'right_only'].drop(columns='_merge')
fn = merged[merged['_merge'] == 'left_only'].drop(columns='_merge')

#union between tp and fp, and tp and fn, for the calculation of precision and recall
tp_fp = pd.concat([tp,fp],ignore_index=True).reset_index(drop=True)
tp_fn = pd.concat([tp,fn],ignore_index=True).reset_index(drop=True)

#evaluating our performance
precision = len(tp.index) / len(tp_fp.index)
recall = len(tp.index) / len(tp_fn.index)

#our precision and recall score
print("Precision: ", precision)
print("Recall: ", recall)

#creating a cleaned dataframe from our results
to_drop = pd.merge(df, res, how='inner', left_on='id', right_on='id2')
df_clean = df.drop(to_drop.id[:])
df_clean.to_csv("restaurants_clean.tsv", index=None, header=True, sep='\t')
df_clean

#the false positives that we obtained from our results
print("False positives:\n", fp, "\nNumber: ", len(fp))

#The false negatives that we obtained from our results
print("\nFalse negatives:\n", fn, "\nNumber: ", len(fn))

''' Uploading everything to pymongo '''
client = pymongo.MongoClient("mongodb+srv://jasonwvh:dmdb@restaurant-in9xr.mongodb.net/test?retryWrites=true&w=majority")
db=client.restaurants

#transformation to dictionary before uploading
df_clean = df_clean.astype(str)
db.restaurants.insert_many(df_clean.to_dict(orient='records'))

#reading from our database
rest = db.restaurants
for document in rest.find():
    print(document)
