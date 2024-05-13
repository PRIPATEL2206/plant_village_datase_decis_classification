# Plant Disease classificaion 

## Discreption
My project is about predicting disease of  `Potato , Pepper Bell , Tomato` plants

### model will efficiently detect disease like
### 1. Potato :
1. Potato Early blight
2. Potato Late blight
3. Potato healthy

### 2. Pepper Bell :
1. Pepper bell Bacterial_spot
2. Pepper bell healthy

### 3. Tomato :
1. Tomato Bacterial spot
2. Tomato Early blight
3. Tomato Late blight
4. Tomato Leaf Mold
5. Tomato Septoria leaf spot
6. Tomato Spider mites Two spotted spider mite
7. Tomato  Target Spot
8. Tomato  Tomato YellowLeaf  Curl Virus
9. Tomato  Tomato mosaic virus 
10. Tomato healthy



## Running GUI
### 1. install python in your system

### 2. Goto main folder where requirements.txt file is their and run   
```cmd
    pip install -r requirements.txt
```
### 3. Then goto views folder by running
```cmd
    cd views
``` 
### 4. run backend fastapi by running
```cmd
    python farmingDataSetApi.py
```
#### Now fast api is runing 

### 5. You can see GUI by opening `views/index.html` file from file explorer 
### or
### 5. You can run using live server in vscode

### 6. Ui will open now you can upload images to that website and choose the plant type and click on predict model will predict the Disease of that plant


### 7. you can also drag and drop images to drop box

### 8. you can use images in dataset folder for testing you can download from  `[dataset](https://www.kaggle.com/datasets/arjuntejaswi/plant-village)`

### 9. This will predict confidence score for all disease to be occur and give the solution accordingly

##
##
## Training
### For training their is one folder called training where all training jupyter files are their fro diffrent models

### You can run this file by opening in vs code or by running command from root folder of this file
```cmd 
    jupyter notebook 
```
Now you can run any training file from training 

##
##
## Ready Models
### All ready models are in models folder