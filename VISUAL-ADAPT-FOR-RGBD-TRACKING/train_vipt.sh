# Training ViPT
python tracking/train.py --script vipt --config deep_rgbd --save_dir ./output --mode multiple --nproc_per_node 2
python tracking/train.py --script vipt --config deep_rgbt --save_dir ./output --mode multiple --nproc_per_node 2
python tracking/train.py --script vipt --config deep_rgbe --save_dir ./output --mode multiple --nproc_per_node 2


# Training ViPT-shaw
#python tracking/train.py --script vipt --config shaw_rgbd --save_dir ./output --mode multiple --nproc_per_node 2
#python tracking/train.py --script vipt --config shaw_rgbt --save_dir ./output --mode multiple --nproc_per_node 2
#python tracking/train.py --script vipt --config shaw_rgbe --save_dir ./output --mode multiple --nproc_per_node 2

nohup python tracking/train.py --script vipt --config deep_rgbd --save_dir ./output --mode single >train.log 2>&1 &
nohup python tracking/test.py ostrack vitb_text_ep300 --dataset tnl2k --threads 12 --num_gpus 1 >train.log 2>&1 &