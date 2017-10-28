import numpy as np
import re
import pandas as pd


train = pd.read_csv('E:\\Users\\jok\\PycharmProjects\\mobicup\\data\\train.csv')
set_userid = set(train['userid'])
set_startloc = set(train['geohashed_start_loc'])
startloc = list(set_startloc)
userid = list(set_userid)
print("starting")
userid_p = []
startloc_p = []
endloc_p = []
for i in range(len(userid)):
    temp1 = []
    temp2 = []
    userid_choose = train[train['userid'] == userid[i]]
    temp1.insert(0, userid[i])
    temp1.extend([float(len(userid_choose))/len(train)])
    userid_p.append(temp1)  # p(userid)
    userid_choose_loc = userid_choose['geohashed_start_loc'].value_counts()
    userid_choose_loc = pd.DataFrame(userid_choose_loc)
    userid_choose_loc = userid_choose_loc.reset_index()
    userid_choose_loc = np.array(userid_choose_loc)
    userid_choose_loc[:, -1] = userid_choose_loc[:, -1]/float(len(userid_choose))
    for j in range(len(userid_choose_loc)):
        temp2.extend(userid_choose_loc[j])
    temp2.insert(0,userid[i])
    startloc_p.append(temp2)  #p(start_loc | userid)
for i in range(len(startloc)):
    temp = []
    startloc_choose = train[train['geohashed_start_loc'] == startloc[i]]
    startloc_choose_endloc = startloc_choose['geohashed_end_loc'].value_counts()
    startloc_choose_endloc = pd.DataFrame(startloc_choose_endloc)
    startloc_choose_endloc = startloc_choose_endloc.reset_index()
    startloc_choose_endloc = np.array(startloc_choose_endloc)
    startloc_choose_endloc[:, -1] = startloc_choose_endloc[:, -1] / float(len(startloc_choose))
    for j in range(len(startloc_choose_endloc)):
        temp.extend(startloc_choose_endloc[j])
    temp.insert(0,startloc[i])
    endloc_p.append(temp)
userid_p = pd.DataFrame(userid_p)
startloc_p = pd.DataFrame(startloc_p)
endloc_p = pd.DataFrame(endloc_p)
startloc_p.to_csv('E:\\Users\\jok\\PycharmProjects\\mobicup\\data\\startloc_p.csv')
userid_p.to_csv('E:\\Users\\jok\\PycharmProjects\\mobicup\\data\\userid_p.csv')
endloc_p.to_csv('E:\\Users\\jok\\PycharmProjects\\mobicup\\data\\endloc_p.csv')


