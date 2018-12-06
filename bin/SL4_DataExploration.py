
# Data Exploration
'''
Description: 
    This file provide some function that can be used for Data Exploration.
Function this file Contains:
    - GenerateHoldoutDB: Used to gather critical class observation from InputDF and preserve it in HoldoutDB.
    - AddObsFromHoldoutDB: Used to get observation from HoldoutDB and mmix these with InputDF.
'''

# ----------------------------------------------- Loading Libraries ----------------------------------------------- #
import os, sys, time, ast
import pandas as pd
import numpy as np
from SL0_GeneralFunc import LevBasedPrint, AddRecommendation, CreateKey


# ---------------------------------------------- DatasetPrimAnalysis ---------------------------------------------- #

def DatasetPrimAnalysis(DF):
    '''
    Function to understand the structure of the dataset
    
    np.isnan(yy).any(), np.isinf(xx).any(), np.isinf(yy).any()
    '''
    # -----------<<<  Setting constant values that are to be used inside function  >>>----------- #
    df_explore = DF.copy()
    
#     CycleType = config['IterationAim']['CycleType']
#     RunInTrain = ast.literal_eval(config['CreateHoldoutDB']['EnableInTrainCycle'])
#     RunInPredict = ast.literal_eval(config['CreateHoldoutDB']['EnableInPredictCycle'])
#     ModuleName = config['Config']['ModuleSettingRuleName']
#     KeyFormat = ast.literal_eval(config['DataProcessing_General']['KeyFormat'])
#     HoldoutDBSavingLoc = config['InputPaths']['CriticalClassHoldoutDB']
#     HDB_SigToPPreserve = float(config['CreateHoldoutDB']['FracOrCntCritClassSigToPreservePerIteration'])
#     HDB_UpperLimitOnDbSize = int(config['CreateHoldoutDB']['MaxObsThatCanBeKeptInDB'])
#     FeatureProcess_Dict = ast.literal_eval(config['DataProcessing_General']['FeaturesProcessing'])
#     FeatureToIgnore = [ key for key in FeatureProcess_Dict.keys() if FeatureProcess_Dict[key]['Usage'] == 'Identification' ]
#     LevBasedPrint('Inside "'+GenerateHoldoutDB.__name__+'" function and configurations for this has been set.',1,1)
    
#     # -------------------------------<<<  Allowed To Execute  >>>-------------------------------- #
    
    
    print('Overall dataset shape :', df_explore.shape)
    
    temp = pd.DataFrame(df_explore.isnull().sum(), columns = ['IsNullSum'])
    temp['dtypes'] = df_explore.dtypes.tolist()
    temp['IsNaSum'] = df_explore.isna().sum().tolist()
    
    temp_cat = temp.loc[temp['dtypes']=='O' ,:]
    if (len(temp_cat) > 0):
        df_cat = df_explore.loc[:,temp_cat.index].fillna('Missing-NA')
        print('Dataset shape containing Qualitative feature :', df_cat.shape)
        temp_cat = temp_cat.join(df_cat.describe().T).fillna('')
        temp_cat['CategoriesName'] = [ list(df_cat[fea].unique()) for fea in temp_cat.index ]
        temp_cat['%Missing'] = [ round((temp_cat['IsNullSum'][i] / max(temp_cat['count']))*100,2) for i in range(len(temp_cat)) ]
        display(temp_cat)
#         print(temp_cat)

    temp_num = temp.loc[((temp['dtypes']=='int') | (temp['dtypes']=='float')),:]
    if (len(temp_num) > 0):
        df_num = df_explore.loc[:,temp_num.index]#.fillna('Missing-NA')
        print('Dataset shape containing Quantitative feature :', df_num.shape)
        temp_num = temp_num.join(df_num.describe().T).fillna('')
        temp_num['%Missing'] = [ round((temp_num['IsNullSum'][i] / max(temp_num['count']))*100,2) for i in range(len(temp_num)) ]
        
        ## Converting float value to readable format
        colsFormatToChange = ['mean', 'std', 'min', '25%', '50%', '75%', 'max']
        temp_num['count'] = [ int(ele) for ele  in temp_num['count'] ] 
        for col in colsFormatToChange:
            st_li = [ '{0:.10f}'.format(ele) for ele in temp_num[col] ] 
            temp_num[col] = [ st[:st.index('.')+4] for st in st_li ]
        display(temp_num)
#         print(temp_num)
    
    if len(temp)!=len(temp_cat)+len(temp_num):
        print("Some columns data is missing b/c of data type")
    
    
    # ---------------------------------------<<<  xyz  >>>--------------------------------------- #

    return temp_cat, temp_num
    # ------------------------------------------------------------------------------------------- #

#     # ------------------------------<<<  returning the result  >>>------------------------------- #
#     LevBasedPrint('Input dataframe shape changed from {} --> {}'.format(str(InputDF.shape), str(DF.shape)), 1)
#     LevBasedPrint('Getting Observation from Holdout DB and Mixing in InputDF | Complete', 1)
#     LevBasedPrint('',1,1)
#     return DF
#     # ------------------------------------------------------------------------------------------- #



# ----------------------------------------------------------------------------------------------------------------- #
# DF = InputRawBalancedDF.copy()
# DF['SID'] = [ str(ele) for ele in DF['SID'] ]

# _, _ = DatasetPrimAnalysis(DF)