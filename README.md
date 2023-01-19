# InkGender: Unveiling the Gender of an Author

## 🚩About
Given a photo of a written document InkGender can predict the gender of the author of this document.
Features used are hinge and chain codes

## :page_with_curl: Dataset
you can find the dataset used here : (https://www.kaggle.com/datasets/essamwisamfouad/cmp23-handwritten-males-vs-females)

## 🏁 How to run
All the scripts are in the `src` folder
### Extracting Features
1. Download the training data you wish to use 
2. the model expects documents written by males to have the letter `M` other than that it is a female that wrote the document
3. Run the extract features script 
 ```bash
 python extractFeatures.py --input [training images directory] --output [desired output feature directory]
 ``` 
4. Deafult input is `Training Data` and deafult output is `Training Features`
 ### Training
 1. Run the following command
 ```bash
 python train.py  --input [features directory]  --output [desired output model directory]
 ```
 2. Default input is `Training Features` and default output is `Model`
 
 ### Testing 
 1. Run the following command
 ```bash
 python predict.py --test [testing data directory] --model [model directory] --output [desired output results directory]
 ```
 2. Default test is `Testing Data`, default model is `Model`, and default ouput is `Testing results`
 
 ### Evaluation
 1. If you have the labels of the testing data you can evaluate the model by doing the following 
 2. place your labels in a text file named `ground_truth.txt` 
 3. run the following command
 ```bash
 python run evaluate.py
 ```
 4. note the results have to be in a folder named `Testing results` 

### Run the whole pipeline at once 
1. You can run the whole pipeline at one using the following command 
2. Please place your data in the default directories as specified above
```bash
python runAll.py
```
3. You will be given two option whether you want to use an `existing model`  or you want to run from the start
4. You can use the existing model option to evaluate your own model and compare the result with our model

## ✨ Contributers


<div align="center" width=1189> 



[![](https://github.com/Ahmedmma72.png?size=150)](https://github.com/Ahmedmma72)
<img src="https://github.com/Thebrownboy.png" alt="drawing" width="150"/>
</div>
