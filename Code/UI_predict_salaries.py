"""
This file contain code for building dropdown table.
"""
import os
import tkinter as tk
from tkinter import ttk
import json
import numpy as np
from keras.models import load_model

class SalaryPredictorApp: 
    
    def __init__(self) -> None:
        self.model = self.get_model()
        self.feature_name, self.columns_name = self.get_JSON_file()
        self.create_UI()

    #load model
    def get_model(self):
        model_path = os.path.join(os.getcwd(), 'model', 'ANN_model', 'ANN_best_model.h5')
        try:
            return load_model(model_path)
        except Exception as e:
            print("Error loading model:", e)

    #load JSON file
    def get_JSON_file(self):
        json_path = os.path.join(os.getcwd(), 'json_file', 'columns.json')
        try:
            with open(json_path, 'r') as json_file:
                json_data = json.load(json_file)
            return json_data["feature_name"], json_data["columns_name"]
        except Exception as e:
            print("Error loading JSON file:", e)

   #design drop down table
    def create_dropdown(self, key, values, row):
        tk.Label(self.root, text=key, bg='#f0f0f0').grid(row=row, column=0, padx=5, pady=7, sticky="e")
        combobox = ttk.Combobox(self.root, values=values)
        combobox.grid(row=row, column=1, padx=10, pady=7)
        combobox.current(0)  # Set default value
        return combobox

    #create UI dropdown table
    def create_UI(self):
        #create tkinter window 
        self.root = tk.Tk()
        self.root.title("Data Scientist Salary Prediction")
        self.root.resizable(False, False) 
        self.root.configure(bg='#f0f0f0')  # Set background color

        table_name = [
            'Experience Level',
            'Employment Type', 
            'Work Models', 
            'Work Year',
            'Company Size',
            'Job Title',
            'Employee Residence',
            'Company Location']  

        #create drop down table 
        self.dropdown_table = []
        for idx, key in enumerate(self.columns_name):
            combobox = self.create_dropdown(table_name[idx], self.columns_name[key], idx)
            self.dropdown_table.append(combobox)

        #create exit button
        exit_button = tk.Button(self.root, text="Exit", command=self.exit_app)
        exit_button.grid(row=len(self.columns_name), column=2, padx=5, pady=10, sticky="e")
        
        #create button for prediction
        predict_button = tk.Button(self.root, text="Predict Salary", bg='#4CAF50', fg='white',font=("Helvetica", 10, "bold"),command=lambda: self.predict_salary())
        predict_button.grid(row=len(self.columns_name), column=0, columnspan=2, sticky="we")

        #assign label for displaying prediction
        self.output_label = tk.Label(self.root, text="", font=("Helvetica", 12, "bold"))
        self.output_label.grid(row=len(self.columns_name)+1, column=0, columnspan=3)

        #run tkinter UI
        self.root.mainloop()
        
    # Exit application
    def exit_app(self):
        self.root.destroy()

    #Predict salary from input
    def predict_salary(self):
        #get data from user
        input_data = [box.get() for box in self.dropdown_table]
        idx_list = [] #record index from user (index come from matching with JSON file)
        #find string name that match with column in json file
        for idx, data in enumerate(input_data):
            try:
                idx = self.feature_name.index(list(self.columns_name.keys())[idx]+'_'+data) # data_frame.index("columns_name") = match index with input string 
            except:
                idx = -1
            idx_list.append(idx)

        #convert index to onehot
        all_index = np.zeros(len(self.feature_name)) #record index after convrt to onehot 
        for index in idx_list:
            if index >= 0:
                all_index[index] = 1
        all_index = all_index.reshape(1,len(self.feature_name))

        ##prediction
        prediction = self.model.predict([all_index])
        self.output_label.config(text=f"Predicted Value: {prediction[0][0]:.2f} USD/Year", fg="green") 
        print(f"Prediction Salary = {prediction[0][0]:.2f} USD/Year") 






