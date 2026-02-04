
#nohup bash reproduce.sh > reproduce_result.log 2>&1 &

# BEST RESULTS on LLaMA
python main_test.py --model meta-llama/Llama-2-7b-chat-hf --data_name boolq --k 100 --alpha 15 --applied_module attention
python main_test.py --model meta-llama/Llama-2-7b-chat-hf --data_name copa --k 16 --alpha 50 --applied_module ffn
python main_test.py --model meta-llama/Llama-2-7b-chat-hf --data_name winogrande --k 2 --alpha 100 --applied_module ffn
python main_test.py --model meta-llama/Llama-2-7b-chat-hf --data_name svamp --k 50 --alpha 8 --applied_module attention
python main_test.py --model meta-llama/Llama-2-7b-chat-hf --data_name mawps --k 80 --alpha 8 --applied_module ffn


# BEST RESULTS on Gemma
python main_test.py --model google/gemma-2-9b-it --data_name boolq --k 8 --alpha 50 --applied_module ffn
python main_test.py --model google/gemma-2-9b-it --data_name copa --k 80 --alpha 50 --applied_module ffn
python main_test.py --model google/gemma-2-9b-it --data_name winogrande --k 64 --alpha 100 --applied_module ffn
python main_test.py --model google/gemma-2-9b-it --data_name svamp --k 4 --alpha 100 --applied_module attention
python main_test.py --model google/gemma-2-9b-it --data_name mawps --k 8 --alpha 50 --applied_module attention


# BEST RESULTS on Qwen
python main_test.py --model Qwen/Qwen3-8B --data_name boolq --k 100 --alpha 10 --applied_module ffn
python main_test.py --model Qwen/Qwen3-8B --data_name copa --k 8 --alpha 20 --applied_module attention
python main_test.py --model Qwen/Qwen3-8B --data_name winogrande --k 100 --alpha 20 --applied_module ffn
python main_test.py --model Qwen/Qwen3-8B --data_name svamp --k 100 --alpha 10 --applied_module attention
python main_test.py --model Qwen/Qwen3-8B --data_name mawps --k 2 --alpha 50 --applied_module attention

