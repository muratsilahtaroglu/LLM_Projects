import os,sys,glob


from generate_qa import multithread_processing,dynamic_thread_processing
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    data_path = os.path.join(current_dir,"datasets/dataset.xlsx")
    predictors_path = os.path.join(current_dir,"_predictors.json")
    version = "V2"
    results_folder = "survey_answers"
    user_data_paths = glob.glob(f"{current_dir}/generate_survey/v2/*")
    completed_user_data_paths = glob.glob(f"{results_folder}/{version}/*")
    completed_users = {os.path.basename(folder) for folder in completed_user_data_paths}  
    user_files = {os.path.basename(file).replace('.xlsx', ''): file for file in user_data_paths}  
    
    missing_user_paths = [path for name, path in user_files.items() if name not in completed_users]
    print(len(user_data_paths)-len(missing_user_paths))
    
    survey_flag = True
    dynamic_thread_processing(missing_user_paths,results_folder, data_path, predictors_path, version,survey_flag=survey_flag)
