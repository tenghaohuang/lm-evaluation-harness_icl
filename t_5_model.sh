DIR="google/t5-xxl-lm-adapt"
#DIR="/fruitbasket/models/t5-small"
if [ -d "$DIR" ]; then
  echo "Using weights in ${DIR}"
else
  mkdir $DIR
#  gsutil cp gs://bigscience/experiment_d/bigbench/finetune-t5-xxl-lm-d4-091621-512/1-112-200/* $DIR
  gsutil cp gs://t5-data/pretrained_models/t5.1.1.lm100k.xxl/* $DIR

fi
CUDA_VISIBLE_DEVICES=0,3 python main.py \
--model t5 \
      --model_args parallelize=True,pretrained=$DIR,batch_size=16 \
--tasks rte,cb,copa,wic,wsc \
--provide_description \
--num_fewshot 32 \
--output_path /fruitbasket/users/tenghao/lm-evaluation-harness/t5_results

CUDA_VISIBLE_DEVICES=0,3 python main.py \
--model t5 \
      --model_args parallelize=True,pretrained=$DIR,batch_size=16 \
--tasks anli_r2,anli_r3 \
--provide_description \
--num_fewshot 50 \
--output_path /fruitbasket/users/tenghao/lm-evaluation-harness/t5_results


#CUDA_VISIBLE_DEVICES=0,3 python main.py \
#--model t5 \
#      --model_args parallelize=True,pretrained=$DIR,batch_size=16 \
#--tasks hellaswag \
#--provide_description \
#--num_fewshot 20 \
#--output_path /fruitbasket/users/tenghao/lm-evaluation-harness/t5_results

#CUDA_VISIBLE_DEVICES=1,2 python main.py \
#	--model t5 \
#        --model_args parallelize=True,pretrained=$DIR,batch_size=16 \
#	--tasks wic \
#	--provide_description \
#	--num_fewshot 32
#
#CUDA_VISIBLE_DEVICES=1,2 python main.py \
#	--model t5 \
#        --model_args parallelize=True,pretrained=$DIR,batch_size=16 \
#	--tasks rte,cb,copa \
#	--provide_description \
#	--num_fewshot 32

#CUDA_VISIBLE_DEVICES=1,2 python main.py \
#	--model t5 \
#        --model_args parallelize=True,pretrained=$DIR,batch_size=16 \
#	--tasks winogrande \
#	--provide_description \
#	--num_fewshot 50
#CUDA_VISIBLE_DEVICES=1,2 python main.py \
#--model t5 \
#      --model_args parallelize=True,pretrained=$DIR,batch_size=16 \
#--tasks wsc \
#--provide_description \
#--num_fewshot 32

#CUDA_VISIBLE_DEVICES=1,2 python main.py \
#--model t5 \
#      --model_args parallelize=True,pretrained=$DIR,batch_size=16 \
#--tasks anli_r1,anli_r2,anli_r3 \
#--provide_description \
#--num_fewshot 50


