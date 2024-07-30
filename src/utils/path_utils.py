import os
import json

def from_file_to_name(file)->str:
    return file.split('/')[-1].split('.')[0]

def get_exp_subfolders(experiment_folder:str)->list:
    return [f.path for f in os.scandir(experiment_folder) if (f.is_dir() and f.path.split('/')[-1].startswith('exp'))]

def get_subfolders(experiment_folder:str)->list:
    return [f.path for f in os.scandir(experiment_folder) if f.is_dir()]

def get_not_exp_subfolders(experiment_folder:str)->list:
    return [f.path for f in os.scandir(experiment_folder) if (f.is_dir() and not f.path.split('/')[-1].startswith('exp'))]

def get_experiment_folder(base_folder:str,
                          task:str,
                          template:str,
                          prompt_parameters:str,
                          model_name:str,
                          generation_mode:str,
                          n_few_shot:int,
                          temperature:float,
                          seed:int)->str:
    if generation_mode == "zero_shot" or generation_mode == "rag":
        folder = os.path.join(base_folder, 
                                        f"task_{task}",
                                        f"template_{template}",
                                        f"prompt_parameters_{prompt_parameters}",
                                        f"model_{model_name.replace('/', '_')}",
                                        f"generation_mode_{generation_mode}",
                                        f"temperature_{temperature}",
                                        f"seed_{seed}")
    elif generation_mode == "few_shot" or generation_mode == "rag_few_shot":
        folder = os.path.join(base_folder, 
                                        f"task_{task}",
                                        f"template_{template}",
                                        f"prompt_parameters_{prompt_parameters}",
                                        f"model_{model_name.replace('/', '_')}",
                                        f"generation_mode_{generation_mode}",
                                        f"n_few_shot_{n_few_shot}",
                                        f"temperature_{temperature}",
                                        f"seed_{seed}")
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
                                                opt.generation_mode,
                                                opt.examples_per_class,
                                                opt.temperature,
                                                opt.seed)
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
                                                opt.examples_per_class,
                                                opt.temperature,
                                                opt.seed)
    os.makedirs(experiment_folder, exist_ok=True)
    
    #enumerate the subfolders related to previous runs
    current_run_number = get_last_run_number(experiment_folder) + 1
    run_folder = os.path.join(experiment_folder, f"run_{current_run_number}")
    os.makedirs(run_folder, exist_ok=False)
    return run_folder

def folder_exists_and_not_empty(folder:str)->bool:
    return os.path.isdir(folder) and os.listdir(folder)

def from_folder_to_accuracy_list(folder:str, top_k_list = None)->list:
    #find all the subfolder in folder:
    subfolders = [f.path for f in os.scandir(folder) if f.is_dir() ]
    #keep only subfolder starting with exp_
    
    subfolders = list(filter(lambda x: x.split('/')[-1].startswith('exp_'), subfolders))
    #map subfolder to the dict conteined in results.json file
    results = []
    if top_k_list:
        #filter subfolders keeping only the ones where the last folder is in the list
        subfolders = list(filter(lambda x: x.split('/')[-1] in top_k_list, subfolders))
    for subfolder in subfolders:
        with open(os.path.join(subfolder, 'results.json')) as f:
            results.append(json.load(f))

    results = [r for r in results if not r['failed']]
    #map results to accuracy
    results = [r['results']['accuracy'] for r in results]
    return results

def from_folder_to_accuracy_std_list(folder:str, top_k_list = None)->list:
    #find all the subfolder in folder:
    subfolders = [f.path for f in os.scandir(folder) if f.is_dir() ]
    #keep only subfolder starting with exp_
    
    subfolders = list(filter(lambda x: x.split('/')[-1].startswith('exp_'), subfolders))
    #map subfolder to the dict conteined in results.json file
    results = []
    if top_k_list:
        #filter subfolders keeping only the ones where the last folder is in the list
        subfolders = list(filter(lambda x: x.split('/')[-1] in top_k_list, subfolders))
    for subfolder in subfolders:
        with open(os.path.join(subfolder, 'results.json')) as f:
            results.append(json.load(f))

    results = [r for r in results if not r['failed']]
    #map results to accuracy
    results = [r['results']['accuracy_std'] for r in results]
    return results

def from_folder_to_success(folder:str)->list:

    with open(os.path.join(folder, 'results.json')) as f:
        data = json.load(f)
    success_rate = data['successes']/data['total']
    return success_rate

def from_folder_to_top_k(folder:str, synthetic_dataset:str)->list:
    with open(folder+synthetic_dataset) as f:
        data = json.load(f)
    elements =  data['top_k_metrics'].keys()
    #keep only elements starting with top_
    elements = list(filter(lambda x: x.startswith('top_'), elements))
    return elements
    
def from_folder_to_subsets(folder:str, synthetic_dataset:str)->list:
    with open(folder+synthetic_dataset) as f:
        data = json.load(f)
    return data['top_k_metrics']["subsets"].keys()

def from_folder_to_top_k_experiments(folder:str, synthetic_dataset:str, top_k_metric:str)->list:
    with open(folder+synthetic_dataset) as f:
        data = json.load(f)
        return data['top_k_metrics'][top_k_metric]["experiments"]
    
def from_folder_to_top_k_subset_experiments(folder:str, synthetic_dataset:str, top_k_metric:str, subset:str)->list:
    with open(folder+synthetic_dataset) as f:
        data = json.load(f)
        return data['top_k_metrics']["subsets"][subset][top_k_metric]["experiments"]

def loadFile_recursive(ext,path=os.getcwd()):
    

    cfiles = []
    for root, dirs, files in os.walk(path):
      for file in files:
        #print(file)
        for i in ext:
            if file.endswith(i):
                cfiles.append(os.path.join(root, file))
    #print(cfiles)
    
    #for i in range(0, len(cfiles)):
    #    print(cfiles[i])
    
    return cfiles

def get_list_of_synth_results(folder, synthetic_datasets_folder):
    results_files = loadFile_recursive(['.json'], os.path.join(folder,synthetic_datasets_folder))
    #keep only the folder removing file

    #keep only the relative part starting from folder on
    results_files = list(map(lambda x: x.split(folder)[1], results_files))
    return results_files

def from_folder_to_top_k_accuracy_std(folder:str, synthetic_dataset:str, top_k_list = None):
     #find all the subfolder in folder:
    subfolders = [f.path for f in os.scandir(folder) if f.is_dir()]
    #keep only subfolder starting with exp_
    subfolders = list(filter(lambda x: x.split('/')[-1].startswith('exp_'), subfolders))
    #map subfolder to the dict conteined in results.json file
    results = []
    if top_k_list:
        #filter subfolders keeping only the ones where the last folder is in the list
        subfolders = list(filter(lambda x: x.split('/')[-1] in top_k_list, subfolders))
    for subfolder in subfolders:
        with open(os.path.join(subfolder + synthetic_dataset, 'results.json')) as f:
            results.append(json.load(f))

    results = [r for r in results if not r['failed']]
    #map results to accuracy
    results = [r['results']['accuracy_std'] for r in results]

    return results

def from_folder_to_top_k_subset__accuracy_std(folder:str, synthetic_dataset:str, subset:str, top_k_list = None):
     #find all the subfolder in folder:
    subfolders = [f.path for f in os.scandir(folder) if f.is_dir()]
    #keep only subfolder starting with exp_
    subfolders = list(filter(lambda x: x.split('/')[-1].startswith('exp_'), subfolders))
    #map subfolder to the dict conteined in results.json file
    results = []
    if top_k_list:
        #filter subfolders keeping only the ones where the last folder is in the list
        subfolders = list(filter(lambda x: x.split('/')[-1] in top_k_list, subfolders))
    for subfolder in subfolders:
        with open(os.path.join(subfolder + synthetic_dataset, 'results.json')) as f:
            results.append(json.load(f))

    results = [r for r in results if not r['failed']]
    #map results to accuracy
    results = [r['subsets'][subset]['results']['accuracy_std'] for r in results]

    return results

def get_run_parameters(run_folder):
    params = dict()
    parts = run_folder.split('/')
    list_elements = ["task", "template", "prompt_parameters", "model", "generation_mode", "n_few_shot", "temperature", "seed"]
    #get the part starting with elements
    for el in list_elements:
        part_list = list(filter(lambda x: x.startswith(el), parts))
        if part_list:
            part = part_list[0]
            params[el] = part.split(f'{el}_')[-1]
        else:

            params[el] = 0

    return params


  