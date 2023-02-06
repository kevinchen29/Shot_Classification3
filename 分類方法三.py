import numpy as np
from sklearn.model_selection import train_test_split
import glob as gb
import cv2
x_100=[]
y_100=[]
#%%
data_path = gb.glob("E://tt//124//*.npz")#'I:\\code\\S\\dection_tree_data1_1\\*.npz'
number=1
for path in data_path:   
    npz_file = np.load(path)
    xn= npz_file['x']
    x_100.extend(xn)     
    for i in range(0,len(xn)):
        y_100.extend([number])
    number=number+1

#%%
X_train, X_test, y_train, y_test = train_test_split(x_100, y_100, test_size=0.3, random_state=93)
from sklearn.ensemble import RandomForestClassifier
#隨機森林
forest=RandomForestClassifier(criterion='entropy', n_estimators=510,n_jobs=-1, random_state=29)#,min_samples_split=6,max_depth=15
forest.fit(X_train,y_train)
y_pred = forest.predict(X_test)
print('Misclassified samples: %d' % (y_test != y_pred).sum())
from sklearn.metrics import accuracy_score
print('Accuracy: %.5f' % accuracy_score(y_test, y_pred))
#GradientBoosting
from sklearn.ensemble import GradientBoostingClassifier
gbm0 = GradientBoostingClassifier(learning_rate=0.1,n_estimators=986,random_state=0,min_samples_split=15,max_depth=10)#
gbm0.fit(X_train, y_train)
y_pred = gbm0.predict(X_test)
print('Misclassified samples: %d' % (y_test != y_pred).sum())
from sklearn.metrics import accuracy_score
print('Accuracy: %.5f' % accuracy_score(y_test, y_pred))
#KNeighbors
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5,algorithm='kd_tree',n_jobs=-1)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print('Misclassified samples: %d' % (y_test != y_pred).sum())
from sklearn.metrics import accuracy_score
print('Accuracy: %.5f' % accuracy_score(y_test, y_pred))
#Voting
from sklearn.ensemble import VotingClassifier
eclf1 = VotingClassifier(estimators=[('forest', forest), ('gbm0', gbm0), ('knn', knn)], voting='hard',n_jobs=-1)
eclf1.fit(X_train, y_train)
y_pred = eclf1.predict(X_test)
print('Misclassified samples: %d' % (y_test != y_pred).sum())
from sklearn.metrics import accuracy_score
print('Accuracy: %.5f' % accuracy_score(y_test, y_pred))
#%% train結果
y_pred = eclf1.predict(x_100)
answer = y_100
count=[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]
for pp in range(0,len(x_100)):
    if y_pred [pp]<1.5 :
       
        if 1==answer[pp]:
            count[0][0]=count[0][0]+1
        elif 2==answer[pp]:
            count[1][0]=count[1][0]+1
        elif 3==answer[pp]:
            count[2][0]=count[2][0]+1
        elif 4==answer[pp]:
            count[3][0]=count[3][0]+1
        else:
            count[4][0]=count[4][0]+1
    elif y_pred [pp]<2.5:                
        if 2==answer[pp]:
            count[1][1]=count[1][1]+1
        elif 1==answer[pp]:
            count[0][1]=count[0][1]+1
        elif 3==answer[pp]:
            count[2][1]=count[2][1]+1
        elif 4==answer[pp]:
            count[3][1]=count[3][1]+1
        else:
            count[4][1]=count[4][1]+1
    elif y_pred [pp]<3.5:        
        if 3==answer[pp]:
            count[2][2]=count[2][2]+1
        elif 2==answer[pp]:
            count[1][2]=count[1][2]+1
        elif 1==answer[pp]:
            count[0][2]=count[0][2]+1
        elif 4==answer[pp]:
            count[3][2]=count[3][2]+1
        else:
            count[4][2]=count[4][2]+1
    elif y_pred [pp]<4.5:        
        if 4==answer[pp]:
            count[3][3]=count[3][3]+1
        elif 2==answer[pp]:
            count[1][3]=count[1][3]+1
        elif 3==answer[pp]:
            count[2][3]=count[2][3]+1
        elif 1==answer[pp]:
            count[0][3]=count[0][3]+1
        else:
            count[4][3]=count[4][3]+1
    else:
        if 5==answer[pp]:
            count[4][4]=count[4][4]+1
        elif 2==answer[pp]:
            count[1][4]=count[1][4]+1
        elif 3==answer[pp]:
            count[2][4]=count[2][4]+1
        elif 1==answer[pp]:
            count[0][4]=count[0][4]+1
        else:
            count[3][4]=count[3][4]+1

#%% test
img_path=[]#//*.png
count=[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]
number=550
while number<560:
    #data_path = gb.glob("I:\\code\\S\\dection_tree_data2\\*.npz")
    npz_file = np.load("E:\\dection_tree_data5_2\\"+str(number)+".npz")# tt4\\2_2
    xn= npz_file['x']
    yn = npz_file['y']
    y_pred=eclf1.predict(xn)

    for ii in range(2,len(y_pred)-3):
        if y_pred[ii-2]==y_pred[ii-1]==y_pred[ii+1]==y_pred[ii+2] and y_pred[ii-1]!=y_pred[ii]:
            y_pred[ii]=y_pred[ii-1]
        elif y_pred[ii-2]==y_pred[ii-1]==y_pred[ii+2]==y_pred[ii+3] and y_pred[ii-1]!=y_pred[ii]:
            y_pred[ii]=y_pred[ii-1]
        
    answer = yn
    for pp in range(0,len(xn)):
        if y_pred [pp]<1.5 :            
            if 1==answer[pp]:
                count[0][0]=count[0][0]+1
            elif 2==answer[pp]:
                count[1][0]=count[1][0]+1
            elif 3==answer[pp]:
                count[2][0]=count[2][0]+1
            elif 4==answer[pp]:
                count[3][0]=count[3][0]+1
            elif 5==answer[pp]:
                count[4][0]=count[4][0]+1
        elif y_pred [pp]<2.5:
            if 2==answer[pp]:
                count[1][1]=count[1][1]+1
            elif 1==answer[pp]:
                count[0][1]=count[0][1]+1
            elif 3==answer[pp]:
                count[2][1]=count[2][1]+1
            elif 4==answer[pp]:
                count[3][1]=count[3][1]+1
            elif 5==answer[pp]:
                count[4][1]=count[4][1]+1
        elif y_pred [pp]<3.5:
            if 3==answer[pp]:
                count[2][2]=count[2][2]+1
            elif 2==answer[pp]:
                count[1][2]=count[1][2]+1
            elif 1==answer[pp]:
                count[0][2]=count[0][2]+1
            elif 4==answer[pp]:
                count[3][2]=count[3][2]+1
            elif 5==answer[pp]:
                count[4][2]=count[4][2]+1
        elif y_pred [pp]<4.5:
            if 4==answer[pp]:
                count[3][3]=count[3][3]+1
            elif 2==answer[pp]:
                count[1][3]=count[1][3]+1
            elif 3==answer[pp]:
                count[2][3]=count[2][3]+1
            elif 1==answer[pp]:
                count[0][3]=count[0][3]+1
            elif 5==answer[pp]:
                count[4][3]=count[4][3]+1
        else:
            if 5==answer[pp]:
                count[4][4]=count[4][4]+1
            elif 2==answer[pp]:
                count[1][4]=count[1][4]+1
            elif 3==answer[pp]:
                count[2][4]=count[2][4]+1
            elif 1==answer[pp]:
                count[0][4]=count[0][4]+1
            elif 4==answer[pp]:
                count[3][4]=count[3][4]+1
    number=number+1