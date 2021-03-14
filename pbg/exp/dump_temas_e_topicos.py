import json 
import sys 
import pickle 
import os



def get_tema_topico(m, temas_path='temas.json'):     
    temas = json.load(open(temas_path)) 
    tema_topico = {} 
    topicos = m.get_topics(15) 
    for tema, topico in sorted(m.map_class_.items()):     
        if tema > 0: 
            tema_topico["{}, tema{}: {}".format(topico, tema, temas[str(tema)]) ] = topicos[topico] 
    return tema_topico 


def get_topicos_sem_tema(m):
    tema_topico=get_tema_topico(m)
    topicos_list=m.get_topics(15)  
    topicos_dict_sem_tema=dict(enumerate(topicos_list)) 
    chaves_com_temas=set([ int(k.split(',')[0])  for k in tema_topico]) 
    for k in list(topicos_dict_sem_tema.keys()): 
        if k in chaves_com_temas: 
            topicos_dict_sem_tema.pop( k )
    return topicos_dict_sem_tema


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Gera topicos sem temas.'
    )
    parser.add_argument('--id_run', nargs='+', type=str, )
    parser.add_argument('--dump_folder', type=str, default='tema_topico')
    args, _=parser.parse_known_args()
    print("ARGS:", args)

    args_dict = args.__dict__
    del args
    
    for id_run in args_dict.get('id_run'):
        try:
            m_path=f'mlruns/0/{id_run}/artifacts/pbg_model_spacy/model.pkl' 
            m=pickle.load(open(m_path, 'rb')) 

            tema_topico = get_tema_topico(m)
            topicos_dict_sem_tema = get_topicos_sem_tema(m)

            DUMP_FOLDER = args_dict.get('dump_folder')
            os.makedirs(DUMP_FOLDER, exist_ok=True)

            json.dump(
                tema_topico,
                open(f'{DUMP_FOLDER}/tema_topico_{id_run}.json', 'w'),
                    indent=4*' ', ensure_ascii=False
            ) 

            json.dump(
                topicos_dict_sem_tema,
                open(f'{DUMP_FOLDER}/topicos_sem_tema_nltk_small_{id_run}.json', 'w'),
                    indent=4*' ', ensure_ascii=False
            ) 
        except Exception as e:
            print(f"[DBG] Unexpected error `{e}` for {id_run}")
            print("[DBG] Gonna keep the script...")