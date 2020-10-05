import os, json
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix

#%% 1. Join these two data sets by “date” and “source_id”, returning all rows from both regardless of whether
#there is a match between the two data sets.
# Read “InterviewData_Cost.csv” and “InterviewData_Rev.csv”.
df1 = pd.read_csv('InterviewData_Cost.csv')
df2 = pd.read_csv('InterviewData_Rev.csv')
result = df1.merge(df2, how = "outer", left_on = ["date", "source_id"], right_on = ["date", "source_id"])

#%% 2. Join these two data sets by “date” and “source_id”, returning only the rows from the “Cost” file that
#have no corresponding date in the “Revenue” file.
res_null = pd.isnull(result["revenue"])
res_not_null = pd.notnull(result["cost"])
result_no_corres = result[res_null & res_not_null]

#%% 3. Using your result from #1:
#a. What are the Top 4 sources (“source_id” values) in terms of total revenue generation across this data set?
result_top_4 = result.groupby("source_id")["revenue"].sum().sort_values(ascending = False)
print(result_top_4.head(4))
#%%
#b. How would you visualize the monthly revenue for those Top 4 sources?
figure = plt.figure(figsize = (20,15))
ax = figure.add_axes([0,0,1,1])
ax.set_ylim([0,1800000])
x_labels = result_top_4[:4].tolist()
ax.set_ylabel("Revenue", fontsize = 20)
ax.set_title("Monthly Revenue for Top 4 Sources", fontsize = 25)   
ax.axhline(result.groupby("source_id")["revenue"].sum().mean(), color='black', linewidth=2)
result_top_4.head(4).plot(kind = 'bar')

#%% 4. build a basic logistic regression model:

df3 = pd.read_csv("InterviewData_Activity.csv")

dummy_genders = pd.get_dummies(df3['gender'], prefix = 'gender')
dummy_metro = pd.get_dummies(df3['metropolitan_area'], prefix = 'metro_area')
dummy_device = pd.get_dummies(df3['device_type'], prefix = 'device')
cols_to_keep = ['active', 'age']
activity_data = df3[cols_to_keep].join(dummy_genders.loc[:, 'gender_M':])
activity_data = activity_data.join(dummy_metro.loc[:, 'metro_area_Birmingham':])

activity_data = activity_data.join(dummy_device.loc[:, 'device_Mobile':])
activity_data = sm.add_constant(activity_data, prepend=False)
explanatory_cols = activity_data.columns[1:]
full_logit_model = sm.GLM(activity_data['active'], activity_data[explanatory_cols], family=sm.families.Binomial())
result = full_logit_model.fit()
print(result.summary())

def confusionMatrix(y, X):
    predicted_values = []
    for value in result.predict(X).tolist():
        if value >= .5:
            predicted_values.append(1)
        else:
            predicted_values.append(0)
            
    tn, fp, fn, tp = confusion_matrix(y,predicted_values).ravel()
    print(tn, fp, fn, tp)
    accuracy = (tp + tn) / (tp + tn + fn + fp)
    precision = (tp) / (tp + fp)
    recall = (tp)/ (tp + fn)
    f1_score = 2 * ((precision * recall)/(precision + recall))
    
    print("Accuracy: ", accuracy, ", Precision :", precision, ", Recall :", recall, ", F-1 Score :", f1_score)
        
confusionMatrix(activity_data['active'], activity_data[explanatory_cols])

#%% 5. Split the data into training and test samples, and build a model over the training data

training_data = activity_data[1:4000]
test_data = activity_data[4001:].copy()
training_logit_model = sm.GLM(training_data['active'], training_data[explanatory_cols], family=sm.families.Binomial())
training_result = training_logit_model.fit()

confusionMatrix(test_data['active'], test_data[explanatory_cols])

#%% 6. This data comes from a subset of userdata JSON blobs stored in our database. Parse out the values(stored in the 
#“data_to_parse” column) into four separate columns. So for example, the four additional
#columns for the first entry would have values of “N”, “U”, “A7”, and “W”. 

df_parse = pd.read_csv("InterviewData_Parsing.csv")
df_parse["data_to_parse"] = df_parse["data_to_parse"].apply(lambda x: eval(str("{" + x).replace("]", "")))

df_parse["One"] = df_parse["data_to_parse"].apply(lambda x: x["value"].split(";")[0])
df_parse["Two"] = df_parse["data_to_parse"].apply(lambda x: x["value"].split(";")[1])
df_parse["Three"] = df_parse["data_to_parse"].apply(lambda x: x["value"].split(";")[2])
df_parse["Four"] = df_parse["data_to_parse"].apply(lambda x: x["value"].split(";")[3])



