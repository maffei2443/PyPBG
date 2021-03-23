# Primeiro vá para o diretório `params` da `run` em questão.
# Depois, basta executar esse script.

for f in $(ls);
    do echo -n "$f --> ";
    echo $(cat $f);
done;
