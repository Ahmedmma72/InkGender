from itertools import chain
import cv2 
import numpy as np 
import glob 
import time 
import os 
from multiprocessing import Process , Lock,Value

def preprocess(img):

    gray = None
    gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
   
    
    # using adaptive thresholding for each image not a static threshold for all images 
    threshold_image = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, 199, 5)
    

    blur = None    # blur the image to remove the noise
    blur=cv2.blur(threshold_image,(3,3));  
       
    return blur


def lbp_feature(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = cv2.LBP(gray)
    print(lbp)
    # lbp_hist = lbp.calcHist([0], [0], None, [256], [0, 256])
    # return lbp_hist




# lbp_feature(img1)









# #very imporatant decrease the number of contours 

# index=[]
# for j in range(len(hierarchy[0])):
#     if hierarchy[0][j][2]==-1:
#         index.append(j)
#         continue
#     if hierarchy[0][hierarchy[0][j][2]][2]==-1:
#         index.append(j)


# # minimizing the number of conouots to be the contours of letters only 
# print(len(index))







# for i in range(len(contours)):

#     cv2.drawContours(img1,contours,i,(0,255,0),1)


# print(contours[1241][1]-contours[1241][0])
# print(contours[1241][0],contours[1241][1])


female_set=[]
male_set=[]

def reading(set):
    for filename in glob.iglob(f"/home/thebrownboy/Desktop/CMP_2023_Third_Year/2th Term/Pattern_Recognition_NN/Project/archive/{set}/{set}/*.jpg"):
        img1=cv2.imread(filename)

        img2=preprocess(img1)

        contours, hierarchy  = cv2.findContours(img2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)


        dir=[0,0,0,0,0,0,0,0]

        chain_code = ""

        chain_code_pair = np.zeros((8,8))


        for i in range(len(contours)):
            chain_code=""
            for j in range (1,len(contours[i]),1):
                if (contours[i][j]-contours[i][j-1]==np.array([[-1,-1]])).all():
                    dir[0]+=1
                    chain_code=chain_code+"0"
                elif (contours[i][j]-contours[i][j-1]==np.array([[0,-1]])).all():
                    dir[1]+=1
                    chain_code=chain_code+"1"
                elif (contours[i][j]-contours[i][j-1]==np.array([[1,-1]])).all():
                    dir[2]+=1
                    chain_code=chain_code+"2"
                elif (contours[i][j]-contours[i][j-1]==np.array([[1,0]])).all():
                    dir[3]+=1
                    chain_code=chain_code+"3"
                elif (contours[i][j]-contours[i][j-1]==np.array([[1,1]])).all():
                    dir[4]+=1
                    chain_code=chain_code+"4"
                elif (contours[i][j]-contours[i][j-1]==np.array([[0,1]])).all():
                    dir[5]+=1
                    chain_code=chain_code+"5"
                elif (contours[i][j]-contours[i][j-1]==np.array([[-1,1]])).all():
                    dir[6]+=1
                    chain_code=chain_code+"6"
                elif (contours[i][j]-contours[i][j-1]==np.array([[-1,0]])).all():
                    dir[7]+=1
                    chain_code=chain_code+"7"

            for k in range(1,len(chain_code),1):
                chain_code_pair[int(chain_code[k-1])][int(chain_code[k])]+=1
                


        #normalization 
        rangeo=np.max(chain_code_pair)-np.min(chain_code_pair)
        chain_code_pair-=np.min(chain_code_pair)
        chain_code_pair=chain_code_pair/rangeo

        dir = np.array(dir).reshape(1,8)
        rangeo=np.max(dir)-np.min(dir)
        dir-=np.min(dir)
        dir=dir/rangeo 

        feature=np.concatenate((chain_code_pair.flatten().reshape([1,64]),dir),axis=1)
        if set=="Males":
            male_set.append(feature)
            print("append in males")
        else:
            female_set.append(feature)
            print("append in females")
        print(len(male_set))
        print(len(female_set))

    if set=="Males":
        print("I am saving our data ")
        np.save("Males.npy",male_set)
    else:
        print("I am saving their data ")

        np.save("Females.npy",female_set)








if __name__=="__main__":
    male_process=Process(target=reading,args=("Males",))
    female_process=Process(target=reading,args=("Females",))
    male_process.start()
    female_process.start()

    male_process.join()
    female_process.join()



male_set=np.load("Males.npy")
female_set=np.load("Females.npy")

males_labels=np.ones([len(male_set),1])
females_labels=np.ones([len(female_set),1])


labels=np.concatenate([males_labels,females_labels])





male_np_arr= np.array(male_set).reshape(len(male_set),72)
female_np_arr= np.array(female_set).reshape(len(female_set),72)


train_data=np.concatenate([male_np_arr,female_np_arr])


print(train_data)
print(train_data.shape)


# print(np.array(data_set).reshape(len(data_set),72))
# print()







# cv2.imshow("contours",img1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

