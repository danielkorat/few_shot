python3.6 -m virtualenv .env
source .env/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
wget https://raw.githubusercontent.com/huggingface/transformers/v4.3.2/examples/language-modeling/run_mlm.py