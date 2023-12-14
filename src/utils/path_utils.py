import os

def from_file_to_name(file)->str:
    return file.split('/')[-1].split('.')[0]

def get_subfolders(experiment_folder:str)->list:
    return [f.path for f in os.scandir(experiment_folder) if f.is_dir()]

def get_experiment_folder(base_folder:str,
                          task:str,
                          template:str,
                          prompt_parameters:str,
                          model_name:str,
                          temperature:float)->str:
    folder = os.path.join(base_folder, 
                                     f"task_{task}",
                                     f"template_{template}",
                                    f"prompt_parameters_{prompt_parameters}",
                                    f"model_{model_name}",
                                    f"temperature_{temperature}")
    #replace - with _ and . with _
    #folder = folder.replace('-', '_').replace('.', '_')
    return folder

def get_last_run_number(experiment_folder:str, default:int = -1)->int:
    subfolders = get_subfolders(experiment_folder)
    runs_numbers = list(map(lambda x: int(x.split('/')[-1].split('_')[1]), subfolders))
    return max(runs_numbers, default=default) 

def get_last_run(opt)->str:
    base_folder = opt.experiments_folder
    task = from_file_to_name(opt.task)
    template = from_file_to_name(opt.template)
    prompt_parameters = from_file_to_name(opt.prompt_parameters)

    experiment_folder = get_experiment_folder(base_folder, 
                                              task, 
                                              template, 
                                              prompt_parameters,
                                                opt.model_name, 
                                                opt.temperature)
    run_number = get_last_run_number(experiment_folder, default=0)
    return os.path.join(experiment_folder, f"run_{run_number}")

def create_folder_for_experiment(opt)->str:
    base_folder = opt.experiments_folder
    os.makedirs(base_folder, exist_ok=True)
    task = from_file_to_name(opt.task)
    template = from_file_to_name(opt.template)
    prompt_parameters = from_file_to_name(opt.prompt_parameters)

    experiment_folder = get_experiment_folder(base_folder, 
                                              task, 
                                              template, 
                                              prompt_parameters,
                                                opt.model_name,
                                                opt.generation_mode, 
                                                opt.temperature)
    os.makedirs(experiment_folder, exist_ok=True)
    
    #enumerate the subfolders related to previous runs
    current_run_number = get_last_run_number(experiment_folder) + 1
    run_folder = os.path.join(experiment_folder, f"run_{current_run_number}")
    os.makedirs(run_folder, exist_ok=False)
    return run_folder