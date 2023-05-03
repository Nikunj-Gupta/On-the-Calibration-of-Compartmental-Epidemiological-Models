all: 
	clear 
	python optim_train.py 

MODELS = SIR_n2 SIRD_n2 SIRVD_n2 
gen_data: 
	for M in $(MODELS); do \
		python model_gen.py \
		--file_json configs/noisy/config_$$M.json \
		--num_sim_days 175 \
		--noise_level 2 \
		--plot 0 \
		--save_data 1;\
		done; 
