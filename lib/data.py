import os

FB2M_KG = '../data/SimpleQuestions_v2/freebase-subsets/freebase-FB2M.txt'
FB5M_KG = '../data/SimpleQuestions_v2/freebase-subsets/freebase-FB5M.txt'

# Get the path relative to the directory this file is in
_directory_path = os.path.dirname(os.path.realpath(__file__))
FB2M_KG = os.path.realpath(os.path.join(_directory_path, FB2M_KG))
FB5M_KG = os.path.realpath(os.path.join(_directory_path, FB5M_KG))

FB2M_KG_TABLE = 'fb_two_kg'
FB5M_KG_TABLE = 'fb_five_kg'
FB2M_NAME_TABLE = 'fb_two_subject_name'
FB5M_NAME_TABLE = 'fb_five_subject_name'
