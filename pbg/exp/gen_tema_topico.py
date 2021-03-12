"""Script para gerar o diretório `tema_topico` com os jsons contendo:
    1 - tópicos relativos aos temas
    2 - tópicos "extras" (não-supervisionado), para além dos temas
"""


import re
import os
import dump_temas_e_topicos as dmp
import json


os.makedirs('tema_topico', exist_ok=True)
lis_json=os.listdir('tema_topico') 
lis_runs=os.listdir('mlruns/0/') 


ids_processadas = set([re.search(r'_([a-z0-9]+).json', i).group(1) for i in lis_json])
ids_runs = set([re.search(r'(?:(^[a-z0-9]+)$|)', i).group(1) for i in lis_runs])     
if None in ids_runs:
    ids_runs.remove(None)
print("processadas:", ids_processadas)
print("ids_runs:", ids_runs)

to_process = [id_run for id_run in ids_runs if id_run not in ids_processadas]

os.system(f"python dump_temas_e_topicos.py --id_run {' '.join(to_process)}")

