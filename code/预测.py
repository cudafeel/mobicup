import numpy as np
import pandas as pd
#predict the test

startloc_p = pd.read_csv('E:\\Python\\mobicup\\data\\startloc_p.csv',low_memory=False)  #read the values in bayesian network
userid_p = pd.read_csv('E:\\Python\\mobicup\\data\\userid_p.csv',low_memory=False)
endloc_p = pd.read_csv('E:\\Python\\mobicup\\data\\endloc_p.csv',low_memory=False)
test = pd.read_csv('E:\\Python\\mobicup\\data\\test.csv')
userid_hotloc = pd.read_csv('E:\\Python\\mobicup\\data\\userid_hotloc.csv',low_memory=False)
staloc_hotloc = pd.read_csv('E:\\Python\\mobicup\\data\\staloc_hotloc.csv',low_memory=False)

test = np.array(test)
predict_test = []
print("start predicting")
for line in test:
     userid_t = startloc_p[startloc_p['0'] == line[1]] #userid
     endloc_t = endloc_p[endloc_p['0'] == line[5]]
     i = 1;j = 1
     temp = []
     startloc_probility = 0
     if userid_t.empty:
          if endloc_t.empty:
               predict_test.append([line[0],'wx4dtmh','wx4dtmh','wx4dtmh'])
          else:
               start_hotloc = staloc_hotloc[staloc_hotloc['0'] == line[5]]
               start_hotloc = np.array(start_hotloc)
               predict_test.append([line[0], start_hotloc[0][2], start_hotloc[0][4], start_hotloc[0][6]])


     else:
          userid_t = np.array(userid_t) #p(geohashed_start_loc | userid)
          endloc_t = np.array(endloc_t) #p(geohashed_end_loc | geohashed_start_loc)
          while(i < len(userid_t)-2):
               if line[5] == userid_t[i+1]:
                    startloc_probility = userid_t[i+2]
                    while(j < len(endloc_t)-2):
                         temp.extend(startloc_probility * endloc_t[j+2])
                         j += 2
                    temp1 = np.argsort(-np.array(temp))
                    predict_test.append([line[0],endloc_t[0][temp1[0]*2-1],endloc_t[0][temp1[1]*2-1],endloc_t[0][temp1[2]*2-1]])
               else:
                    i +=2
          if startloc_probility == 0:
               userid_hot = userid_hotloc[userid_hotloc['0'] == line[1]]
               userid_hot = np.array(userid_hot)
               predict_test.append([line[0],userid_hot[0][2],userid_hot[0][4],userid_hot[0][6]])
predict_test = pd.DataFrame(predict_test)
predict_test.to_csv('E:\\Python\\mobicup\\data\\predict.csv')


