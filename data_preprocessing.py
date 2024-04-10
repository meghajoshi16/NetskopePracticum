import pandas as pd 
import random

class preprocess(object):
    def __init__(self, import_csv):
        self.csv_name_import = import_csv
        self.func_role_unique_csv = "func_role_unique.csv"
        self.job_level_unique_csv = "job_level_unique.csv"


    def load_data(self):
        historical = pd.read_csv(self.csv_name_import)
        historical["Title"] = historical["Title"].str.lower()
        historical = historical.dropna()
        historical = historical[historical.Title != "** no longer with company **"]
        historical = historical[historical.Title != "-"]
        historical["Title"] = historical["Title"].str.replace('[^a-zA-Z ]', ' ')
        historical['Title'] = historical['Title'].str.strip()
        historical.drop(historical[historical["Job Function"] == "Unknown"].index, inplace=True)
        historical.drop(historical[historical["Job Level"] == "Unknown"].index, inplace=True)
        return(historical)

    def preprocess(self, historical_df):
        combos = historical_df.groupby(['Title','Job Role','Job Function', 'Job Level']).size().reset_index().rename(columns={0:'count'}) 
        combos = combos.sort_values(by = ["Title"])
        combos_duplicated = combos['Title'].duplicated(keep = False)
        tf_df = pd.DataFrame({'index':combos_duplicated.index, 'TF':combos_duplicated.values, "Title": combos.Title})
        repeat_index = list(tf_df[tf_df['TF'] == True]["index"])
        repeat_df = combos.iloc[repeat_index] 
        repeat_titles_list =  repeat_df["Title"].unique()

        for title in list(repeat_titles_list):
            mini_repeat_df = repeat_df[repeat_df['Title'] == title]
            count_total = mini_repeat_df["count"].sum()
            portion_list = []
            for i in range(len(mini_repeat_df)):
                portion_list.append(mini_repeat_df.iloc[i]['count'])
            portion_list = list(portion_list/count_total)
            max_val = max(portion_list)
            max_index = portion_list.index(max_val)
            if max_val >= 0.5:
                final_row = mini_repeat_df.iloc[max_index]
            if max_val < 0.5: 
                randomIndex = random.randint(0, len(mini_repeat_df)-1)
                final_row = mini_repeat_df.iloc[randomIndex]
            final_dict = dict(final_row)
            historical_df.loc[historical_df['Title'] == title, 'Job Role'] = final_dict['Job Role']
            historical_df.loc[historical_df['Title'] == title, 'Job Function'] = final_dict['Job Function']
            historical_df.loc[historical_df['Title'] == title, 'Job Level'] = final_dict['Job Level'] 
        return(historical_df)
            
    def to_csv(self, historical_df):
        remap_func_role = pd.read_csv(self.func_role_unique_csv)
        remapped_historical = pd.merge(historical_df,remap_func_role , how="left", on=["Job Function", "Job Role"])
        remapped_historical = remapped_historical.drop('count', axis=1)
        remap_level = pd.read_csv(self.job_level_unique_csv)
        remapped_historical = pd.merge(remapped_historical,remap_level , how="left", on=["Job Level"])
        remapped_historical.drop_duplicates('Title', inplace=True)
        remapped_historical.reset_index(inplace=True)
        remapped_historical = remapped_historical.drop(columns=['index'])
        return(remapped_historical)

