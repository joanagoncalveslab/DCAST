import os
import sys
import subprocess

path2this = os.path.dirname(os.path.abspath(__file__)).split('/')
for i, folder in enumerate(path2this):
    if folder.lower() == 'diversepsuedolabeling':
        project_path = '/'.join(path2this[:i + 1])
sys.path.insert(0, project_path)
from src import config
print(f'{config.ROOT_DIR}: {os.path.exists(config.ROOT_DIR)}')
print(f'{config.ROOT_DIR / "src" / "analysis" / "find_missing.py"}: {os.path.exists(config.ROOT_DIR / "src" / "analysis" / "find_missing.py")}')
print(f'myfile: {os.path.exists("/tudelft.net/staff-bulk/ewi/insy/DBL/ytepeli/DiversePsuedoLabeling/results_fsd_test_nb_imb_ss8/random/mnist/random(True|30)_mnist(0.3|0.2|0.2|0.7)_DST-BRRF1(th=85|kb=30|mi=100|vb=False|b=ratio).csv")}')



print(f'Current processes:')
s = subprocess.getstatusoutput(f'squeue -h -O JobID -j $jobid -u ytepeli')
job_ids = s[1].replace(' ','').split('\n')
job_ids.sort()
status_dict = {}
for job_id in job_ids:
    out_file = config.ROOT_DIR / 'out' / f'slurm-{job_id}.out'
    #subprocess.getstatusoutput(f'head -3 {out_file}')
    try:
        status_dict[subprocess.getstatusoutput(f'head -3 {out_file}')[1].split('/ytepeli/')[-1]] = job_id
        #print(f"{job_id}: {subprocess.getstatusoutput(f'head -3 {out_file}')[1].split('/ytepeli/')[-1]}")
    except:
        print(f"{job_id} could not be opened")



bias_size = 30
#main_folder = '/tudelft.net/staff-bulk/ewi/insy/DBL/ytepeli/DiversePsuedoLabeling'
ds_classes = {'breast_cancer': 2, 'wine_uci2': 2, 'mushroom': 2, 'mnist': 10, 'rice': 2, 'fire': 2, 'spam': 2, 'adult': 2, 'raisin': 2, 'pumpkin': 2, 'pistachio': 2}
for ds, class_size in ds_classes.items():
    th=97
    if ds=='mnist':
        th=85
    print(f'\n\n{ds} \n')
    bias_dict = {'None': 'None()', 
                 'dirichlet': f'dirichlet({bias_size*class_size})', 
                 'hierarchyy_0.9': f'hierarchyy(True|{bias_size}|0.9)',
                 'hierarchyy_0.7': f'hierarchyy(True|{bias_size}|0.7)',
                 'hierarchyy_0.5': f'hierarchyy(True|{bias_size}|0.5)',
                 'joint': f'joint()',
                 'random': f'random(True|{bias_size})'
                }
    
    folder_bias_dict = {'results_test_nb_imb_ss8': ['None', 'dirichlet', 'hierarchyy_0.9', 'hierarchyy_0.7', 'hierarchyy_0.5', 'joint', 'random'],
                   'results_fsd_test_nb_imb_ss8': ['dirichlet', 'hierarchyy_0.9',  'joint', 'random'],
                   'results_extra_test_nb_imb_ss8': ['dirichlet', 'hierarchyy_0.9', 'joint', 'random'],
                   'results_nn_test_nb_imb_fin_cw3_ss8': ['None', 'dirichlet', 'hierarchyy_0.9', 'hierarchyy_0.7', 'hierarchyy_0.5', 'joint', 'random'],
                   'results_fsdnn_test_nb_imb_fin_cw3_ss8': ['dirichlet', 'hierarchyy_0.9',  'joint', 'random'],
                   'results_nn_extra_test_nb_imb_fin_cw3_ss8': ['dirichlet', 'hierarchyy_0.9', 'joint', 'random'],
                   'results_kmm_rf_test_nb_imb_ss8': ['dirichlet', 'hierarchyy_0.9', 'joint', 'random'],
                   'results_kmm_nn_test_nb_imb_ss8': ['dirichlet', 'hierarchyy_0.9', 'joint', 'random'],
                   'results_lr_test_nb_imb_ss8': ['dirichlet', 'hierarchyy_0.9', 'joint', 'random'],
                   'results_lr_fsd_test_nb_imb_ss8':  ['None','dirichlet', 'hierarchyy_0.9', 'joint', 'random'],
                   'results_lr_extra_test_nb_imb_ss8':  ['dirichlet', 'hierarchyy_0.9', 'joint', 'random'],
                  }
    
    for folder, biases in folder_bias_dict.items():
        for bias_name in biases:
            bias_out_name = bias_dict[bias_name]
            if 'hierarchyy' in bias_name:
                bias_name = 'hierarchyy'
            folder_dict = {'results_test_nb_imb_ss8': f'{bias_out_name}_{ds}(0.3|0.2|0.2|0.7)_DST-BRRF1(th={th}|kb={class_size*3}|mi=100|vb=False|b=ratio)_es=True',
                           'results_nn_test_nb_imb_fin_cw3_ss8': f'{bias_out_name}_{ds}(0.3|0.2|0.2|0.7)_DST-BRRF1(th={0.9}|kb={class_size*3}|mi=100|vb=False|b=ratio)_es=True',
                           'results_lr_test_nb_imb_ss8':  f'{bias_out_name}_{ds}(0.3|0.2|0.2|0.7)_DST-BRRF1(th={97}|kb={class_size*3}|mi=100|vb=False|b=ratio)',
                           'results_lr_fsd_test_nb_imb_ss8':  f'{bias_out_name}_{ds}(0.3|0.2|0.2|0.7)_DST-BRRF1(th={0.9}|kb={class_size*3}|mi=100|vb=False|b=ratio)_es=True',
                           'results_kmm_rf_test_nb_imb_ss8': f'{bias_out_name}_{ds}(0.3|0.2|0.2|0.7)_KMM-BRRF1(th=rbf|vb=False)',
                           'results_kmm_nn_test_nb_imb_ss8': f'{bias_out_name}_{ds}(0.3|0.2|0.2|0.7)_KMM-BRRF1(th=rbf|vb=False)',
                           'results_fsdnn_test_nb_imb_fin_cw3_ss8': f'{bias_out_name}_{ds}(0.3|0.2|0.2|0.7)_DST-BRRF1(th={0.9}|kb={class_size*3}|mi=100|vb=False|b=ratio)',
                           'results_fsd_test_nb_imb_ss8': f'{bias_out_name}_{ds}(0.3|0.2|0.2|0.7)_DST-BRRF1(th={th}|kb={class_size*3}|mi=100|vb=False|b=ratio)',
                           'results_extra_test_nb_imb_ss8': f'{bias_out_name}_{ds}(0.3|0.2|0.2|0.7)_DST-BRRF1(th={0.9}|kb={class_size*3}|mi=100|vb=False|b=ratio)',
                           'results_nn_extra_test_nb_imb_fin_cw3_ss8': f'{bias_out_name}_{ds}(0.3|0.2|0.2|0.7)_DST-BRRF1(th={0.9}|kb={class_size*3}|mi=100|vb=False|b=ratio)',
                           'results_lr_extra_test_nb_imb_ss8': f'{bias_out_name}_{ds}(0.3|0.2|0.2|0.7)_DST-BRRF1(th={0.9}|kb={class_size*3}|mi=100|vb=False|b=ratio)_es=False',
                          }
            out_csv_loc = config.ROOT_DIR / folder / bias_name / ds / f'{folder_dict[folder]}.csv'
            out_log_loc = config.ROOT_DIR / folder / bias_name / ds / f'{folder_dict[folder]}.log'
            if not os.path.exists(out_csv_loc):
                try:
                    print(f'{out_csv_loc}: {status_dict[str(out_log_loc).split("/ytepeli/")[-1]]}')
                except:
                    print(f'{out_csv_loc}: Not found ')

