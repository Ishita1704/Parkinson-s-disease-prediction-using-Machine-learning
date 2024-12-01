Evaluation metrics are used to measure the performance of machine learning models. Python, with libraries like scikit-learn, provides tools to compute these metrics efficiently. Here's an overview of some common evaluation metrics and how to calculate them in Python:

![image](https://github.com/user-attachments/assets/84c6c0fb-7f9c-4f8a-9a7b-7af984d5a971)
![image](https://github.com/user-attachments/assets/167ba60d-5bf9-4678-804c-aa48a741d7c1)
![image](https://github.com/user-attachments/assets/47199b48-c9f8-4d74-a62d-18a8f86cb1ff)
![image](https://github.com/user-attachments/assets/d8b772e0-3156-4541-b037-2e96ec9826e6)
![image](https://github.com/user-attachments/assets/fbb31641-d727-464c-ac15-376dbfe4d147)
3. Ranking/Probability Metrics
Area Under the ROC Curve (AUC-ROC):
![image](https://github.com/user-attachments/assets/06f7978f-8dc3-4aa2-8e4b-4914b89c8a7a)
![image](https://github.com/user-attachments/assets/4fcdc6cd-3baf-4e8d-8686-858fca61ff7d)
OUTPUT:
Parkinson's disease data
![image](https://github.com/user-attachments/assets/61e4d86f-dd35-4c40-8e51-bfde444c06b5)
Dataframe size:
![image](https://github.com/user-attachments/assets/f634ea5c-0e32-4591-a3c7-1028655d4f9a)
Defining numerical & categorical columns
![image](https://github.com/user-attachments/assets/cb9e673b-dced-4e6a-9fb6-08af8de4bab5)
Missing Value Presence in different columns of DataFrame
![image](https://github.com/user-attachments/assets/609938f1-52a1-4754-9290-d838d2a76e73)
Summary Statistics of numerical features for DataFrame 
![image](https://github.com/user-attachments/assets/29eaa7f6-6883-40f6-bc0b-6f887ab4bd43)
Data status
![image](https://github.com/user-attachments/assets/74fe5019-2856-4f85-bb11-a737bb8283a6)
Separating the data and labels:
x
![image](https://github.com/user-attachments/assets/9d96ea8e-a384-49d5-86c6-0d547263ea33)
y
![image](https://github.com/user-attachments/assets/322a16be-a65f-4106-9fd1-c6a22ca4fcb6)
scaler = StandardScaler()
scaler.fit(X)
![image](https://github.com/user-attachments/assets/6efcf2ec-c012-482c-8de4-1e08f1528da1)
standardized_data = scaler.transform(X)
standardized_data![image](https://github.com/user-attachments/assets/f8556e12-53d2-4ab0-9e01-6b8441cbeb97)
X = standardized_data
X
![image](https://github.com/user-attachments/assets/760b58e6-9e36-4285-8469-f94ac0ba9709)
Models= LogisticRegression, SVC, DecisionTreeClassifier, RandomForestClassifier
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []
![image](https://github.com/user-attachments/assets/1764c4b2-e5a8-443e-924b-aa7ae1e17f0c)
Evaluation:
![image](https://github.com/user-attachments/assets/df14df98-948b-4da0-bc57-82608fa2e475)






