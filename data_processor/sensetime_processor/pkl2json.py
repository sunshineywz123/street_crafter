import pickle
import json
import argparse


def pkl2json(pkl_file, json_file):
    # Step 1: 读取 pkl 文件
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)

    # Step 2: 尝试将其转为 JSON（前提是它是可序列化的）
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4, default=str)

    print("转换完成：data.json")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl_file', type=str, default='/nas/home/yanyunzhi/waymo/training')
    parser.add_argument('--json_file', type=str, default='./test_data/')
    args = parser.parse_args()
    
    pkl_file = args.pkl_file
    json_file = args.json_file

    pkl2json(pkl_file, json_file)


if __name__ == '__main__':
    main()