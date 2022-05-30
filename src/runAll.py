import os

# run all the scripts

# check if he wants to run the extract features script
print("1. run the whole pipeline")
print("2. use a saved model")
option = int(input("Enter your option: "))
while(option != 1 and option != 2):
    option = int(input("Enter your option: "))


if(option == 1):
    os.system("python extractFeatures.py")
    os.system("python train.py")
    os.system("python predict.py")
    os.system("python evaluate.py")
else:
    os.system("python predict.py")
    os.system("python evaluate.py")
