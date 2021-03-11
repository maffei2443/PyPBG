import json 
import sys 
import pickle 
 
m_path='mlruns/0/6b9f3496ff2e403cb4dfff6e0b71c0e3/artifacts/pbg_model_spacy/model.pkl' 
m=pickle.load(open(m_path, 'rb')) 

def dump_tema_topico(m, temas_path='temas.json'):     
    temas = json.load(open(temas_path)) 
    tema_topico = {} 
    topicos = m.get_topics(15) 
    for tema, topico in sorted(m.map_class_.items()):     
        if tema > 0: 
            tema_topico["{}, tema{}: {}".format(topico, tema, temas[str(tema)]) ] = topicos[topico] 
    json.dump(tema_topico, open('tema_topico_nltk_small.json', 'w'), indent=4*' ', ensure_ascii=False) 
    return tema_topico 


tema_topico=dump_tema_topico(m)
topicos_list=m.get_topics(15)  
topicos_dict=dict(enumerate(topicos_list))  
topicos_dict_sem_tema=topicos_dict.copy() 
chaves_com_temas=set([ int(k.split(',')[0])  for k in tema_topico]) 
for k in list(topicos_dict_sem_tema.keys()): 
  if k in chaves_com_temas: 
    topicos_dict_sem_tema.pop( k )

json.dump(
    tema_topico,
    open('topicos_sem_tema_nltk_small.json', 'w'),
        indent=4*' ', ensure_ascii=False
) 
