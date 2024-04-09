"""

This file show training progress before getting the model.

"""
import os
import pandas as pd 
import matplotlib.pyplot as plt
from Preprocessing_function import DataCleaning
from ANN_function import ANN_model
from linear_model_function import find_best_linear_model

def main():
    #TODO: Data Preprocessing 
    #read file
    file_path  = os.path.join(os.getcwd(), 'dataset', 'data_science_salaries.csv')
    try:
        with open(file_path, 'rb') as file:
            df0 =  pd.read_csv(file_path)
    except Exception as e:
        print("Error loading model:", e)

    #drop some table
    print("In Dataframe_1:")
    print("Use currency in USD units >> drop unrelated tables.")
    df1 = df0.drop(["salary", "salary_currency"], axis="columns")
    print("[Drop 'salary' and 'salary_currency' tables]\n")
    
    #drop NA value
    print("In Dataframe_2:")
    print(f"Total N/A value = {sum(df1.isna().sum())}")
    df2 = df1.dropna()
    print("[Drop NA value]\n")

    #group duplicate nmae in job title
    df3 = df2.copy()
    data_cleaning = DataCleaning()
    print("In Dataframe_3:")
    print("[Group duplicated job title name]")
    df3["job_title"] = df3["job_title"].apply(data_cleaning.group_job_title)
    print(f"Job tile before grouping = {len(df2['job_title'].unique())}")
    print(f"Job tile after grouping = {len(df3['job_title'].unique())}\n")

    #create other category
    print("In Dataframe_4:")
    print("[Group small data point to 'Other' category]\n")
    group_col_name_list = ["job_title", "employee_residence", "company_location"]
    df4 = data_cleaning.create_other_category(df3, group_col_name_list)

    #remove outlier in salary with Z-score
    print("In Dataframe_5:")
    print("[Remove outlier in 'Salary' with Z-score]")
    df5 = data_cleaning.remove_outlier_with_ZSCORE(df4, "salary_in_usd")
    #Plot [Before VS After remove Outlier] graph
    plt.figure()
    plt.hist(df4["salary_in_usd"], bins=150, range=[min(df4["salary_in_usd"]), max(df4["salary_in_usd"])], color='red', label='with outlier')
    plt.hist(df5["salary_in_usd"], bins=150, range=[min(df4["salary_in_usd"]), max(df4["salary_in_usd"])], color='green',label='without outlier')
    plt.title('Before VS After remove Outlier')
    plt.legend()
    print("\n")

    #Show relation between Experience Level and Salary before remove outlier
    print("In Dataframe_6:")
    print("Entry level shouldn't have salary more than Executive-level")
    print("[Remove outlier in 'Experience Level' with Z-score]")
    plt.figure()
    plt.scatter(df5["experience_level"], df5["salary_in_usd"])
    plt.title('Relation between Experience Level and Salary')
    #remove outlier in experience level with Z-score
    df6 = pd.DataFrame()
    experience_level_list = df5["experience_level"].unique()
    for exp in experience_level_list:
        experience_no_outlier =  data_cleaning.remove_outlier_with_ZSCORE(df5[df5["experience_level"] == exp], "salary_in_usd")
        df6 = pd.concat([df6, experience_no_outlier])
    df6["work_year"] = df6["work_year"].apply(lambda x: str(x))
    df6 = df6.reset_index(drop=True)
     #Show relation between Experience Level and Salary after remove outlier
    plt.figure()
    plt.scatter(df6["experience_level"],df6["salary_in_usd"])
    plt.title('After remove outlier in Experience Level')
    print("\n")

    #TODO: Training Model
    print("In Dataframe_7:")
    print("Dataframe_7 is final dataframe.")
    print("\n")
    print("Waiting for model training...")

    #train linear model
    df7_linear = pd.get_dummies(df6, dtype=float, drop_first=True)
    X_linear = df7_linear.drop("salary_in_usd", axis="columns") 
    y_linear = df7_linear["salary_in_usd"]
    df_result, linear_models = find_best_linear_model(X_linear,y_linear)


    #train ANN model
    df7 = pd.get_dummies(df6, dtype=float, drop_first=False)
    X = df7.drop("salary_in_usd", axis="columns") 
    y = df7["salary_in_usd"]
    ann_model = ANN_model()
    cros_val_score, trained_models =  ann_model.find_best_ANN_model(X,y)


    return df_result, linear_models, cros_val_score, trained_models


if __name__ == "__main__":
    df_result, linear_models, cros_val_score, trained_models = main()
    print("Finish Training and Saving Model.")


