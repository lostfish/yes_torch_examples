
function train_model()
{
    word_level=$1
    batch_size=$2
    model_type=$3
    device=$4
    seq_len=$5
    num_layers=$6

    vocab_size=256
    model_file=./results/model.char.seq${seq_len}.${model_type}

    if [ $word_level -gt 0 ];then
        vocab_size=36221 #TODO
        model_file=./results/model.word.seq${seq_len}.${model_type}.v2
        python main.py --mode train \
            --train_path /data1/aclImdb/dataset_train.pkl \
            --valid_path /data1/aclImdb/dataset_valid.pkl \
            --max_epoch 100 \
            --batch_size $batch_size \
            --embed_size 128 \
            --hidden_size 128 \
            --output_size 2 \
            --word_level \
            --vocab_size $vocab_size \
            --seq_len $seq_len \
            --model_file $model_file \
            --model_type $model_type \
            --device $device \
            --num_layers $num_layers
    else
        python main.py --mode train \
            --train_path /data1/aclImdb/dataset_train.pkl \
            --valid_path /data1/aclImdb/dataset_valid.pkl \
            --max_epoch 100 \
            --batch_size $batch_size \
            --embed_size 128 \
            --hidden_size 128 \
            --output_size 2 \
            --vocab_size $vocab_size \
            --seq_len $seq_len \
            --model_file $model_file \
            --model_type $model_type \
            --device $device \
            --num_layers $num_layers
    fi
    
}

# model_type: 1-4 -> cnn, lstm, transformer, dpcnn

function run()
{
## char-level
train_model 0 512 1 0 2048 1 > log.char.1 &
train_model 0 512 2 1 2048 1 > log.char.2 &
train_model 0 192 3 2 2048 1 > log.char.3 &
train_model 0 512 4 3 2048 10 > log.char.4 &

# word-level
train_model 1 512 1 4 256 1 > log.word.1 &
train_model 1 512 2 5 256 1 > log.word.2 &
train_model 1 192 3 6 256 1 > log.word.3 &
train_model 1 512 4 7 256 7 > log.word.4 &
}

run
