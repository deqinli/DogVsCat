import os
import csv




imageDir_train = "D:/Dataset/DogVsCat/test/"

trainList = os.listdir(imageDir_train)
print(len(trainList))

AbsFileName = ""
label = -1

train_csv = "./image/test.csv"



with open(train_csv,"w",newline='') as f:
    writer = csv.writer(f)

    for strName in trainList:
        category = strName.split(".")[0]
        AbsFileName = imageDir_train + strName
        if "cat"==category:
            label = 0
        elif "dog"==category:
            label = 1
        # print(AbsFileName," ",label)

        writer.writerows([[AbsFileName,label]])
f.close()