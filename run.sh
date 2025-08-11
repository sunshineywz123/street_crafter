# python render.py --config /iag_ad_01/ad/yuanweizhong/huzeyu/street_crafter/configs/waymo_val_049.yaml mode diffusion


python sensetime_processor/sensetime_converter.py --root_dir /iag_ad_01/ad/yuanweizhong/datasets/senseauto/2024_09_08_07_53_23_pathway_pilotGtParser --save_dir /iag_ad_01/ad/yuanweizhong/huzeyu/street_crafter/data --process_list pose calib image track dynamic
python sensetime_processor/img_processor.py
python waymo_processor/waymo_get_moge_pcd.py --data_dir /iag_ad_01/ad/yuanweizhong/huzeyu/street_crafter/data 