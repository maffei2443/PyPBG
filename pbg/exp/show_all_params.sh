for id in $(ls "mlruns/0"); do
    # If is NOT a directory, skips
    if [[ ! -d "mlruns/0/$id" ]];
    then
        continue ;
    fi
    echo "RUN_ID: $id"
    prf="mlruns/0/$id/params"
    for p in $(ls $prf); do
        printf "\t%s: " "${p##*/}";
        echo $(cat "$prf/$p");
    done;
done

