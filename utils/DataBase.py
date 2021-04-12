from math import fabs
from os.path import split
from re import sub
from utils.tools import addWordsToJieba, splitSentence
import ujson
import os
from utils.config import DATASET
import jieba
from io import BytesIO, StringIO

attraction_db_path = "attraction_db.json"
hotel_db_path = "hotel_db.json"
metro_db_path = "metro_db.json"
restaurant_db_path = "restaurant_db.json"
taxi_db_path = "taxi_db.json"

EntityIndex = 0
AttrsDictIndex = 1


#SPO_index satified MEMTOKEN
SUBJECT_INDEX=0
PREDICATE_INDEX=1
OBJECT_INDEX=2

"""
(subject-predicate-object(predicateInfo))
(entity-predicate-predicateInfo)
(subject-name-entity) 

name is kind of predicate
entity is object
"""

SUBJECT_KEY = "领域"

ENTITIES_KEY = "名称"

SUBJECTS = ["景点", "酒店", "餐馆", "地铁", "出租"]

def getDictfromDataBase(filepath: str):
    abspath = os.path.join(os.getcwd(), "data", DATASET, "database", filepath)
    database_dict = None
    with open(abspath,encoding='utf-8') as f:
        database_dict = ujson.load(f)
    return database_dict

# equals
# attraction_db = getDictfromDataBase(attraction_db_path)
# hotel_db = getDictfromDataBase(hotel_db_path)
# metro_db = getDictfromDataBase(metro_db_path)
# restaurant_db = getDictfromDataBase(restaurant_db_path)
# taxi_db = getDictfromDataBase()
dbs = [getDictfromDataBase(path) for path in iter((
    attraction_db_path, hotel_db_path, metro_db_path, restaurant_db_path, taxi_db_path))]


# ChooseDataBaseBySubjectName = {SUBJECTS[i]: db for i,db in enumerate(dbs)}
ChooseDataBaseBySubjectName = dict()
for i, each in enumerate(SUBJECTS):
    ChooseDataBaseBySubjectName.setdefault(each,dbs[i])

PREDICATES = {}

PREDICATES = {eachSubject: [key for key in ChooseDataBaseBySubjectName[
    eachSubject][0][AttrsDictIndex].keys()] for eachSubject in SUBJECTS}

# for eachSubject in SUBJECTS:
#     database = ChooseDataBaseBySubjectName[]

ENTITIES = []
ENTITIES_belongs_SUBJECTS={}


def initPredicate(dbs: tuple):
    for eachSubject in SUBJECTS:
        database = ChooseDataBaseBySubjectName[eachSubject]
        attrsObj = database[0][AttrsDictIndex]
        PREDICATES.setdefault(eachSubject,[])
        for key in attrsObj.keys():
            PREDICATES[eachSubject].append(key)

def initEntitiesAndEntities_belongs(dbs: tuple):
    for index , database in enumerate(dbs):
        for item in database:
            ent = item[EntityIndex]
            ENTITIES.append(ent)
            ENTITIES_belongs_SUBJECTS.setdefault(ent,SUBJECTS[index])


initPredicate(dbs)
initEntitiesAndEntities_belongs(dbs)

# 避免jieba将数据集词拆分
# 读入却分词无效，jieba背锅
# dict_path = os.path.join(os.getcwd(), 'data', 'crossWOZ', 'dict.txt')
# if os.path.isfile(dict_path):
#     with open(dict_path, "r+", encoding="utf8") as file:
#         for each in SUBJECTS:
#             file.writelines(' 3 n \n'.join(PREDICATES[each]))
#         file.writelines(' 3 n \n'.join(SUBJECTS))
#         file.writelines(' 3 n \n'.join(ENTITIES))
#         jieba.load_userdict(file)

for each in SUBJECTS:
    addWordsToJieba(PREDICATES[each])
addWordsToJieba(SUBJECTS)
addWordsToJieba(ENTITIES)


# def getSubjectByEntityThroughDBs(dbs: tuple, ent: str) -> str:
#     for database in dbs:
#         for item in database:
#             if item[EntityIndex] is ent:
#                 return item[AttrsDictIndex][SUBJECT_KEY]
#     return None

def getSubjectByEntity(ent: str) -> str:
    return ENTITIES_belongs_SUBJECTS[ent]

def getAttrsByEntityThroughDBs(dbs: tuple, ent: str) -> dict:
    for database in dbs:
        for item in database:
            if item[EntityIndex] is ent:
                return item[AttrsDictIndex]
    return None

def getAttrsByEntity(ent: str) -> dict:
    database = ChooseDataBaseBySubjectName[ENTITIES_belongs_SUBJECTS[ent]]
    for item in database:
        if item[EntityIndex] == ent:
            return item[AttrsDictIndex]
    return None


def getEntitesBySPO(subject: str, predicate: str, predicateInfo: str):
    database = ChooseDataBaseBySubjectName[subject]
    entities = []
    # entities = [item[EntityIndex] if item[AttrsDictIndex][predicate] is predicateInfo else None for item in database]
    for item in database:
        if item[AttrsDictIndex][predicate] is predicateInfo:
            entities.append(item[EntityIndex])
    return entities if len(entities)>0 else None


def getEntitesBySubject(subject: str)->list:
    ents = []
    for item in ChooseDataBaseBySubjectName[subject]:
        ents.append(item[EntityIndex])
    return ents if len(ents) else None

def getEntityAttrs(ent:str):
    database = ChooseDataBaseBySubjectName[ENTITIES_belongs_SUBJECTS[ent]]
    for item in database:
        if item[EntityIndex] is ent:
            return item[AttrsDictIndex]        

def getEntitesAttrsBySubjectAndPredicate(subject: str, predicate: str)->dict:
    database = ChooseDataBaseBySubjectName[subject]
    # ENTITIES_Attrs = {item[EntityIndex]: {key: item[AttrsDictIndex][key]
    #     for key in item[AttrsDictIndex].keys()} if  item is predicate else None for item in database}
    ENTITIES_Attrs = {}
    for item in database:
        for key in item[AttrsDictIndex].keys():
            if key is predicate:
                ENTITIES_Attrs.setdefault(item[EntityIndex],item[AttrsDictIndex])
    return ENTITIES_Attrs if len(ENTITIES_Attrs) else None

# def getEntitiesBySubjectAndInformPredicate(subject: str, predicate: str,inform_predicate) -> dict:
#     database = ChooseDataBaseBySubjectName[subject]
#     ENTITIES = []
#     for item in database:
#         if item[AttrsDictIndex][predicate] is inform_predicate:
#             ENTITIES.append(item[EntityIndex])
#     return ENTITIES if len(ENTITIES) else None

def findEntities(splitWords:list):
    ents = []
    for word in splitWords:
        if ENTITIES.__contains__(word):
            ents.append(word)
    return ents if len(ents) else None

def findPredicatesBySubject(splitWords:list,subject:str):
    predicates=[]
    for word in splitWords:
        if PREDICATES[subject].__contains__(word):
            predicates.append(word)
    return predicates if len(predicates) else None 

def findPredicatesByEnt(splitWords:list,ent:str):
    predicates = []
    for word in splitWords:
        if PREDICATES[ENTITIES_belongs_SUBJECTS[ent]].__contains__(word):
            predicates.append(word)
    return predicates if len(predicates) else None

def findSubjects(splitWords:list):
    subjects = []
    for word in splitWords:
        if SUBJECTS.__contains__(word):
            subjects.append(subjects)
    return subjects if len(subjects) else None
    

def compareInfoEqual(wordlist, keys):
    for word in wordlist:
        for key in keys:
            if word is key:
                return True
    return False

def wordListFindRequestPredicateInfo(wordlist, old_ents)->dict:
    result =None
    userWants = {}
    subjects = findSubjects(wordlist)


    inform_predicate = [findPredicatesBySubject(wordlist,subject) for subject in subjects]


    ents = findEntities(wordlist)
    if ents is None:
        ents = old_ents

    # if subjects:
    #     ents = getEntitesBySubject()
    #     for ent in ents:
    #         ents_info_list.append(ent)
        
    if ents and inform_predicate:
        userWants.setdefault(inform_predicate, [])
        for ent in ents:
            attrs = getAttrsByEntity(ent)
            for word in wordlist:
                for key, val in enumerate(attrs):
                    if word is val:
                        userWants[inform_predicate].append(ent[inform_predicate])

    elif subjects and inform_predicate:
        # user need ent
        if ents:
            userWants.setdefault(ENTITIES_KEY,[])
            for ent in ents:
                # attrs = getAttrsByEntity(ent)
                predicates = PREDICATES[ENTITIES_belongs_SUBJECTS(ent)]
                if compareInfoEqual(wordlist, predicates):
                    userWants[ENTITIES_KEY].append(ent)
        else:
            ents = getEntitesBySubject(
                subjects)
            userWants.setdefault(ENTITIES_KEY, ents)
    
    return userWants if len(userWants) else None

def getPredicateInfoByEntityThroughDBs(dbs: tuple, ent: str, predicate: str) -> str:
    for database in dbs:
        for item in database:
            if item[EntityIndex] is ent:
                return item[AttrsDictIndex][predicate]
    return None


def generateAllSPO(user_split_words,sys_answer_sentence=None):
    SPO_list = []
    contains_entities = []
    if sys_answer_sentence:
        for word in splitSentence(sys_answer_sentence):
            if word in ENTITIES:
                contains_entities.append(word)
    for word in user_split_words:
        if word in ENTITIES:
            contains_entities.append(word)
    for word in contains_entities:
        database = ChooseDataBaseBySubjectName[ENTITIES_belongs_SUBJECTS[word]]
        for item in database:
            if item[EntityIndex] == word:
                for predicate,object in item[AttrsDictIndex].items():
                    if isinstance(object,list):
                        for slice in object:
                            SPO_list.append([word,predicate,slice]) # tuple
                    elif object is not None:
                        SPO_list.append([word,predicate,object])
    return SPO_list


def patternSubject(wordList):
    for index , word in enumerate(wordList):
        if word in SUBJECTS:
            return word
    return None

def patternPredicateWithSubject(wordList,subject):
    for index, word in enumerate(wordList):
        if word in subject:
            return PREDICATES[subject]
    return None

def patternEntity(wordList):
    for index , word in enumerate(wordList):
        if word in ENTITIES:
            return word
    return None
