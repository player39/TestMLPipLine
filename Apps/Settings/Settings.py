
import os

# RootPath = os.path.abspath(os.path.dirname(os.getcwd()))
RootPath = os.path.abspath(os.path.join(os.getcwd(), '../..'))
ProjectPath = os.path.dirname(os.getcwd())

# Model File Name
strModelFileName = 'HEDModelV2_2_SGD_GradientTape_L1'

# DataSet
strDataSetRootPath = RootPath + '/Apps/Resource/DataSet'
# listDSPath = ['/FastTrainFrame']
dictDS = {'FastTrainFrame': 'Image1/500'}


