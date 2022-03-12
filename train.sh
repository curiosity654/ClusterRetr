for (( i = 1; i <= 5; i++ )); do
    echo exp$i
    seed=$RANDOM
    if [ ! -d "checkpoints/$3/$1/random_$seed" ];then
        # echo $1 checkpoints/$3/$1/random_$seed
        python $1 --num_hashing $2 --kld_lambda $3 --seed $seed --savedir checkpoints/$4/$2/random_$seed
    fi
    # echo checkpoints/$3/$1/random_$seed
done