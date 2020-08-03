python ../plant_triplet_net/a__neural_net_training.py  -i ./a__args_AlexNet.json
python ../plant_triplet_net/b__neural_net_inference.py  -i ./b__args.json
python ../plant_triplet_net/c__clustering_task_training.py  -i ./c__args.json
python ../plant_triplet_net/d__clustering_task_inference.py  -i ./d__args.json
python ../plant_triplet_net/e__plot_everything.py  -i ./e__args.json
