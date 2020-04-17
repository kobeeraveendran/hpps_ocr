import os
import argparse
import random
import shutil

parser = argparse.ArgumentParser()

parser.add_argument("--dir", type = str, default = "~/Documents/attention_ocr/")
parser.add_argument("--n", type = int, default = 100)
args = parser.parse_args()

files = os.listdir(args.dir)

assert(len(files) >= args.n)

sampled_list = random.sample(files, args.n)

os.makedirs("sample", exist_ok = True)

for file in sampled_list:
    path = os.path.join(os.path.abspath("output"), file)
    shutil.copy(path, os.getcwd() + "/sample/" + file)