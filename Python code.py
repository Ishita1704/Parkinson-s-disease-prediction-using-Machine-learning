import numpy as np
import pandas as pd

# for model buidling
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
parkinsons_disease_data = pd.read_csv('/content/parkinsons.csv')
parkinsons_disease_data
print('The size of Dataframe is: ', parkinsons_disease_data.shape)
print('-'*100)
print('The Column Name, Record Count and Data Types are as follows: ')
parkinsons_disease_data.info()
print('-'*100)
# Defining numerical & categorical columns
numeric_features = [feature for feature in parkinsons_disease_data.columns if parkinsons_disease_data[feature].dtype != 'O']
categorical_features = [feature for feature in parkinsons_disease_data.columns if parkinsons_disease_data[feature].dtype == 'O']

# print columns
print('We have {} numerical features : {}'.format(len(numeric_features), numeric_features))
print('\nWe have {} categorical features : {}'.format(len(categorical_features), categorical_features))
print('Missing Value Presence in different columns of DataFrame are as follows : ')
print('-'*100)
total=parkinsons_disease_data.isnull().sum().sort_values(ascending=False)
percent=(parkinsons_disease_data.isnull().sum()/parkinsons_disease_data.isnull().count()*100).sort_values(ascending=False)
pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print('Summary Statistics of numerical features for DataFrame are as follows:')
print('-'*100)
parkinsons_disease_data.describe()
parkinsons_disease_data['status'].value_counts() # status is target variable
# separating the data and labels
X = parkinsons_disease_data.drop(columns = ['name', 'status'], axis=1) # Feature matrix
y = parkinsons_disease_data['status'] # Target variable
X
y
scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)
standardized_data
X = standardized_data
X
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=45)
print(X.shape, X_train.shape, X_test.shape)
print(y.shape, y_train.shape, y_test.shape)
models = [LogisticRegression, SVC, DecisionTreeClassifier, RandomForestClassifier]
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

for model in models:
    classifier = model().fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    accuracy_scores.append(accuracy_score(y_test, y_pred))
    precision_scores.append(precision_score(y_test, y_pred))
    recall_scores.append(recall_score(y_test, y_pred))
    f1_scores.append(f1_score(y_test, y_pred))
  import numpy as np
import pandas as pd

# for model buidling
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
parkinsons_disease_data = pd.read_csv('/content/parkinsons.csv')
parkinsons_disease_data
print('The size of Dataframe is: ', parkinsons_disease_data.shape)
print('-'*100)
print('The Column Name, Record Count and Data Types are as follows: ')
parkinsons_disease_data.info()
print('-'*100)
# Defining numerical & categorical columns
numeric_features = [feature for feature in parkinsons_disease_data.columns if parkinsons_disease_data[feature].dtype != 'O']
categorical_features = [feature for feature in parkinsons_disease_data.columns if parkinsons_disease_data[feature].dtype == 'O']

# print columns
print('We have {} numerical features : {}'.format(len(numeric_features), numeric_features))
print('\nWe have {} categorical features : {}'.format(len(categorical_features), categorical_features))
print('Missing Value Presence in different columns of DataFrame are as follows : ')
print('-'*100)
total=parkinsons_disease_data.isnull().sum().sort_values(ascending=False)
percent=(parkinsons_disease_data.isnull().sum()/parkinsons_disease_data.isnull().count()*100).sort_values(ascending=False)
pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print('Summary Statistics of numerical features for DataFrame are as follows:')
print('-'*100)
parkinsons_disease_data.describe()
parkinsons_disease_data['status'].value_counts() # status is target variable
# separating the data and labels
X = parkinsons_disease_data.drop(columns = ['name', 'status'], axis=1) # Feature matrix
y = parkinsons_disease_data['status'] # Target variable
X
y
scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)
standardized_data
X = standardized_data
X
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=45)
print(X.shape, X_train.shape, X_test.shape)
print(y.shape, y_train.shape, y_test.shape)
models = [LogisticRegression, SVC, DecisionTreeClassifier, RandomForestClassifier]
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

for model in models:
    classifier = model().fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    accuracy_scores.append(accuracy_score(y_test, y_pred))
    precision_scores.append(precision_score(y_test, y_pred))
    recall_scores.append(recall_score(y_test, y_pred))
    f1_scores.append(f1_score(y_test, y_pred))
  classification_metrics_df = pd.DataFrame({
    "Model": ["Logistic Regression", "SVM", "Decision Tree", "Random Forest"],
    "Accuracy": accuracy_scores,
    "Precision": precision_scores,
    "Recall": recall_scores,
    "F1 Score": f1_scores
})

classification_metrics_df.set_index('Model', inplace=True)
classification_metrics_df
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
file_path = '/parkinsons.csv'  # Update this path if necessary
data = pd.read_csv(file_path)

# Prepare data
X = data.drop(columns=['name', 'status'])  # Drop irrelevant columns
y = data['status']  # Target column

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate evaluation metrics
evaluation_matrix = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "F1 Score": f1_score(y_test, y_pred),
    "Confusion Matrix": confusion_matrix(y_test, y_pred)
}

# Display the evaluation matrix
print("Evaluation Matrix:")
for metric, value in evaluation_matrix.items():
    print(f"{metric}: {value}")
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["No Parkinson's", "Parkinson's"],
            yticklabels=["No Parkinson's", "Parkinson's"])

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
import matplotlib.pyplot as plt

# Plot histograms for each feature in the dataset
data.hist(bins=20, figsize=(20, 15), edgecolor='black')
plt.suptitle("Histograms of Dataset Features", fontsize=20)
plt.show()import matplotlib.pyplot as plt

# Calculate the distribution of the 'status' column
status_counts = data['status'].value_counts()

# Plot the pie chart
plt.figure(figsize=(8, 8))
plt.pie(status_counts, labels=["Parkinson's", "No Parkinson's"], autopct='%1.1f%%', startangle=90, colors=['skyblue', 'salmon'])
plt.title("Distribution of Parkinson's Status in Dataset")
plt.show()
import matplotlib.pyplot as plt
import seaborn as sns

# Count the values in the 'status' column
status_counts = data['status'].value_counts()

# Plot a bar graph
plt.figure(figsize=(8, 6))
sns.barplot(x=status_counts.index, y=status_counts.values, palette="viridis")
plt.title("Count of Parkinson's vs. No Parkinson's Cases")
plt.xlabel("Status (0 = No Parkinson's, 1 = Parkinson's)")
plt.ylabel("Count")
plt.xticks([0, 1], ["No Parkinson's", "Parkinson's"])
plt.show()
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset (update the path if necessary)
file_path = '/content/parkinsons.csv'
data = pd.read_csv(file_path)

# Set the plot size
plt.figure(figsize=(20, 15))

# Plot box plots for each feature, excluding the 'name' and 'status' columns
data_features = data.drop(columns=['name', 'status'])
sns.boxplot(data=data_features, palette="Set3")

# Add plot title and labels
plt.title("Box Plot of Each Feature")
plt.xticks(rotation=90)
plt.ylabel("Values")
plt.show()
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset (update the path if necessary)
file_path = '/content/parkinsons.csv'
data = pd.read_csv(file_path)

# Define features to plot
x_feature = 'MDVP:Fo(Hz)'  # Frequency of fundamental frequency
y_feature = 'MDVP:Fhi(Hz)'  # Maximum fundamental frequency

# Plot scatter plot
plt.figure(figsize=(10, 6))
scatter = plt.scatter(data[x_feature], data[y_feature], c=data['status'], cmap='viridis', alpha=0.7)
plt.colorbar(scatter, label='Status (0 = No Parkinson\'s, 1 = Parkinson\'s)')
plt.xlabel(x_feature)
plt.ylabel(y_feature)
plt.title(f"Scatter Plot of {x_feature} vs. {y_feature}")
plt.show()
import pandas as pd

# Load the dataset
data = pd.read_csv('/content/parkinsons.csv')  # Adjust the path if needed

# Define features (X) and target (y)
X = data.drop(columns=['name', 'status'])  # Drop irrelevant columns
y = data['status']  # Target variable

# Continue with the rest of the script
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict probabilities for the positive class
y_probs = model.predict_proba(X_test)[:, 1]

# Compute the AUC-ROC score
auc_score = roc_auc_score(y_test, y_probs)
print(f"AUC-ROC Score: {auc_score:.2f}")

# Plot the ROC curve
fpr, tpr, _ = roc_curve(y_test, y_probs)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}')
plt.plot([0, 1], [0, 1], 'r--')  # Diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()


