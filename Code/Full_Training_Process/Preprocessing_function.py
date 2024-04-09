"""

This file contain all function for doing [Preprocessing process] for [Data Science Job Salaries 2020 - 2024] dataset

"""
#lits of grouped job title
class DataCleaning:
    
    def __init__(self) -> None:
        """
        list of duplicated data
        """
        self.job_title_groups = {
            "Data Scientist": ["Data Science", "Data Science Engineer"],
            "Lead Data Scientist": ["Data Science Lead", "Data Scientist Lead"],
            "Director of Data Science": ["Data Science Director"],
            "Financial Data Analyst": ["Finance Data Analyst"],
            "Data Analyst": ["Data Analytics Engineer"],
            "Lead Data Analyst": ["Data Analyst Lead", "Data Analytics Lead"],
            "BI Analyst": ["Business Intelligence Analyst"],
            "BI Data Analyst": ["Business Intelligence Data Analyst", "Business Data Analyst"],
            "BI Developer": ["Business Intelligence Developer"],
            "Machine Learning Engineer": ["ML Engineer"],
            "Computer Vision Engineer": ["Computer Vision Software Engineer"],
            "Machine Learning Researcher": ["Machine Learning Research Engineer"],
            "ETL Engineer": ["ETL Developer"] 
            }

    # function to map job title to their group
    def group_job_title(self, job_title):
        """
        #Group duplicate [job title] name from duplicate [job title] list
        input = list [job title] from job_title_list() function
        output = category(key) name of that job title
        """
        for group, titles in self.job_title_groups.items():
            if job_title in titles:
                return group
        return job_title 


    # function to create other category if data point < 10
    def create_other_category(self, df, group_col_name_list):
        """
        #Gruop very small data point and name it "Other" category
        #default threshold to group < 10 data point
        input = your dataframe, specific column name in that dataframe
        output = your dataframe with grouped data
        """
        for column_name in group_col_name_list:
            count_column = df.groupby(column_name)[column_name].agg('count').sort_values(ascending=False)
            less_than_10 = count_column[count_column <= 10]
            grouped_column_name = f"grouped_{column_name}"
            df[grouped_column_name] = df[column_name].apply(lambda x: 'Other' if x in less_than_10 else x)
            df = df.drop(column_name, axis="columns")
        return df


    #remove outlier with Z-score
    def remove_outlier_with_ZSCORE(self, dataframe, col_name):
        """
        #Remove outlier from data with Z-score technique.
        #default threshold of outlier = 3
        #1. 68% of the data points lie between   +/- 1 standard deviation.
        #2. 95% of the data points lie between   +/- 2 standard deviation
        #3. 99.7% of the data points lie between +/- 3 standard deviation
        input = your dataframe, specific column name in that dataframe
        output = your dataframe without outlier data
        """
        dataframe = dataframe.copy()
        col_mean = dataframe[col_name].mean()
        col_std = dataframe[col_name].std()
        dataframe["z_score"] = (dataframe[col_name] - col_mean) / col_std
        dataframe_no_outlier = dataframe[(dataframe["z_score"].abs() < 3)]
        return dataframe_no_outlier.drop(["z_score"], axis="columns")
    
   
 
