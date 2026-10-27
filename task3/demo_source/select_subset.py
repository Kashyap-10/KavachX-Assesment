from pathlib import Path
import csv,json,collections,random,urllib.request,concurrent.futures,hashlib,time,shutil
repo=Path('outputs/KavachX-Assesment');target=repo/'alerts/task3_demo';target.mkdir(parents=True,exist_ok=True)
source=repo/'task3/demo_source';source.mkdir(parents=True,exist_ok=True)
chosen=[]
for original,split,per_class in [('train','calibration',10),('test','test',15)]:
 groups={0:collections.defaultdict(list),1:collections.defaultdict(list)}
 for p in sorted((Path('work/birdsai/annotations')/original).glob('*.csv')):
  frames=collections.defaultdict(list)
  for raw in csv.reader(p.open()):
   try:row=list(map(float,raw[:10]))
   except ValueError:continue
   frames[int(row[0])].append(row)
  for label in [0,1]:
   fs=sorted(f for f,rows in frames.items() if int(any(r[6]==1 for r in rows))==label)
   if not fs:continue
   for fraction in [.25,.5,.75]:
    frame=fs[min(len(fs)-1,int(len(fs)*fraction))]
    groups[label][p.stem.split('_')[0]].append(dict(sequence=p.stem,frame=frame,rows=frames[frame],annotation=p))
 for label in [0,1]:
  keys=sorted(groups[label]);random.Random(42+label).shuffle(keys)
  positions={k:0 for k in keys}
  selected=[]
  while len(selected)<per_class:
   advanced=False
   for group in keys:
    pool=groups[label][group];pos=positions[group]
    if pos<len(pool):
     item=pool[pos];positions[group]+=1;advanced=True
     if any(s['sequence']==item['sequence'] and s['frame']==item['frame'] for s in selected):continue
     selected.append({**item,'source_group':group,'label':label,'split':split,'original_partition':original})
     if len(selected)==per_class:break
   if not advanced:raise ValueError('Not enough source frames')
  chosen+=selected
base='https://storage.googleapis.com/public-datasets-lila/'
for i,item in enumerate(chosen):
 directory='conservationdrones/v01/TrainReal' if item['original_partition']=='train' else 'conservationdrones/v01/conservationdrones-testset/TestReal'
 seq=item['sequence'];frame=item['frame'];filename=f'{item["split"]}_{seq}_{frame:010d}.jpg'
 item['filename']=filename;item['url']=base+directory+'/images/'+seq+'/'+f'{seq}_{frame:010d}.jpg'
 item['annotation_url']=base+directory+'/annotations/'+seq+'.csv'
 item['annotation_file']='task3/demo_source/annotations/'+item['original_partition']+'/'+seq+'.csv'
 dest=repo/item['annotation_file'];dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(item['annotation'],dest)
 item['annotation_sha256']=hashlib.sha256(dest.read_bytes()).hexdigest()
 item.pop('annotation')
def fetch(item):
 p=target/item['filename']
 if not p.exists():
  for attempt in range(3):
   try:
    raw=urllib.request.urlopen(item['url'],timeout=30).read();p.write_bytes(raw);break
   except OSError:
    if attempt==2:raise
    time.sleep(1)
 item['sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
 return item
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 chosen=list(pool.map(fetch,chosen))
manifest={'dataset':'BIRDSAI real nighttime aerial thermal data','source_page':'https://lila.science/datasets/conservationdrones/','author_source':'https://sites.google.com/view/elizabethbondi/dataset','license':'CDLA-Permissive-1.0','license_url':'https://cdla.io/permissive-1-0/','citation':'Bondi et al., BIRDSAI: A Dataset for Detection and Tracking in Aerial Thermal Infrared Videos, WACV 2020','substitute':True,'label_semantics':'Human presence proxy, not authorization or intent. Positive if any original box has class=1; negative if annotated animals are present and no human box is present.','selection':'Seed 42, recording-group round robin over annotated frame quartiles. 20 TrainReal calibration images (10/10), 30 TestReal evaluation images (15/15). No synthetic data, ground-truth crop, image enhancement or model-score selection.','limitations':['Thermal aerial imagery differs from fixed visible/IR CCTV.','Frames within a recording are correlated; site/individual independence is not established.','Rain, shadows, insects and ordinary authorized humans are not a validated negative set.'],'samples':chosen}
(source/'manifest.json').write_text(json.dumps(manifest,indent=2))
fields=['filename','label','split','source_group','sequence','frame','negative_kind','sha256']
with (target/'labels.csv').open('w',newline='') as file:
 w=csv.DictWriter(file,fieldnames=fields);w.writeheader()
 for item in chosen:w.writerow({**{k:item[k] for k in fields if k in item},'negative_kind':'animal_only' if item['label']==0 else ''})
print('Prepared',len(chosen),'real images')
for split in ['calibration','test']:
 rows=[s for s in chosen if s['split']==split]
 print(split,collections.Counter(r['label'] for r in rows),'groups',sorted({r['source_group'] for r in rows}))
