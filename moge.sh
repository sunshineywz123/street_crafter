#!/bash/sh
source activate
conda activate /iag_ad_01/ad/yuanweizhong/miniconda/streetcrafter
cd data_processor
python sensetime_processor/sensetime_converter.py --root_dir /iag_ad_01/ad/yuanweizhong/datasets/senseauto/2024_09_08_07_53_23_pathway_pilotGtParser --save_dir /iag_ad_01/ad/yuanweizhong/huzeyu/sc/data 
python sensetime_processor/img_processor.py --root_dir /iag_ad_01/ad/yuanweizhong/datasets/senseauto/2024_09_08_07_53_23_pathway_pilotGtParser --save_dir /iag_ad_01/ad/yuanweizhong/huzeyu/sc/data
python sensetime_processor/sensetime_get_lidar_pcd.py --root_dir /iag_ad_01/ad/yuanweizhong/datasets/senseauto/2024_09_08_07_53_23_pathway_pilotGtParser --save_dir /iag_ad_01/ad/yuanweizhong/huzeyu/sc/data 
python sensetime_processor/sensetime_render_lidar_pcd.py --data_dir /iag_ad_01/ad/yuanweizhong/huzeyu/sc/data --save_dir /iag_ad_01/ad/yuanweizhong/huzeyu/sc/data --delta_frames 10 --cams 0 --shifts 0
conda activate /iag_ad_01/ad/yuanweizhong/miniconda/vggt
python waymo_processor/waymo_get_moge_pcd.py --data_dir /iag_ad_01/ad/yuanweizhong/huzeyu/sc/data
cd ../data
rm -r lidar/background
rm -r lidar/actor
mv moge/background lidar/background
mv moge/actor lidar/actor
conda activate /iag_ad_01/ad/yuanweizhong/miniconda/streetcrafter
cd ..
python render.py --config /iag_ad_01/ad/yuanweizhong/huzeyu/sc/configs/sensetime_val.yaml mode diffusion
