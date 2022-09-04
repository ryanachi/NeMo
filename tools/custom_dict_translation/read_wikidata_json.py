import json
import argparse
from tqdm import tqdm
import mmap

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_filepath", help="source language", required=True)
parser.add_argument("-o", "--output_filepath")
args = parser.parse_args()

en_2_de = {}

def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines

with open(args.input_filepath, 'r') as f_in:
    for line in tqdm(f_in, total=get_num_lines(args.input_filepath)):
        data = json.loads(line, encoding='utf-8')
        # print(f"data is: {data}")

        if 'en_sitelink' not in data or 'de_sitelink' not in data:
            continue
        else:
            en_2_de[data['en_sitelink']] = data['de_sitelink']

json_object = json.dumps(en_2_de, indent=4) 
with open(args.output_filepath, 'w') as f_out:
    f_out.write(json_object)