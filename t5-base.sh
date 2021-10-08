python main.py \
	--model t5 \
	--model_args device='cpu',pretrained='../t5-base' \
	--tasks lambada \
	--provide_description \
	--num_fewshot 2

