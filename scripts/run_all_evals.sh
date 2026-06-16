source /home/thomas/tpu-2026/venvs/tunix/bin/activate
python scripts/evaluate.py --baseline --preset greedy
python scripts/evaluate.py --experiment-name baseline --step 0 --preset greedy
python scripts/evaluate.py --experiment-name improvement-1 --step 2000 --preset greedy
python scripts/evaluate.py --experiment-name improvement-2 --step 2000 --preset greedy