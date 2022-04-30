python main.py --model t5 --model_args pretrained=/fruitbasket/models/bigscience/T0_3B --device cuda:3 --task rte --num_fewshot 32
python main.py --model t5 --model_args pretrained=/fruitbasket/models/bigscience/T0_3B --device cuda:3 --task wic --num_fewshot 32
python main.py --model t5 --model_args pretrained=/fruitbasket/models/bigscience/T0_3B --device cuda:3 --task winogrande --num_fewshot 50
python main.py --model t5 --model_args pretrained=/fruitbasket/models/bigscience/T0_3B --device cuda:3 --task copa --num_fewshot 32
python main.py --model t5 --model_args pretrained=/fruitbasket/models/bigscience/T0_3B --device cuda:3 --task cb --num_fewshot 32
python main.py --model t5 --model_args pretrained=/fruitbasket/models/bigscience/T0_3B --device cuda:3 --task hellaswag --num_fewshot 20