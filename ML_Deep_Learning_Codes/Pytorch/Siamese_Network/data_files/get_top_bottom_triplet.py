import json
import os

data_files_index = {}
f = open('bam_filename.txt')
for i, line in enumerate(f):
    data_files_index[line.strip()] = i

f = open('top_K.json', 'rb')
s = f.read()
j = json.loads(s)
f.close()

data_files_top = {}
data_files_bottom = {}

for image in j:
    l = j[image]
    image = 'bam_subset_2_0/'+image.strip()
    data_files_top[image] = l

print("Top neighbours dict created")

f = open('bottom_K.json', 'rb')
s = f.read()
j = json.loads(s)
f.close()

for image in j:
    l = j[image]
    image = 'bam_subset_2_0/'+image.strip()
    data_files_bottom[image] = l

print("Bottom neighbours dict created")

