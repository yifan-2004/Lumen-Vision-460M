#!/usr/bin/env python3
"""Rebuild the editable architecture / training figure from its JSON layout."""
import html
import json
import re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
spec = json.loads((ROOT / 'assets/model-layout.json').read_text())
parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1800" height="930" viewBox="0 0 1800 930" role="img" aria-labelledby="title desc">',
'<title id="title">'+html.escape(spec['title'])+'</title><desc id="desc">'+html.escape(spec['description'])+'</desc>',
'<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#536579"/></marker></defs>',
'<rect width="1800" height="930" fill="#ffffff"/>',
'<style>text{font-family:Arial,Helvetica,sans-serif;fill:#172b42}.title{font-size:36px;font-weight:700}.eyebrow{font-size:15px;letter-spacing:2px;fill:#657c92}.panel{font-size:20px;font-weight:700}.label{font-size:19px;font-weight:600}.small{font-size:15px;fill:#52687e}.foot{font-size:14px;fill:#64768a}.wire{fill:none;stroke:#536579;stroke-width:2}</style>']
def text(x,y,value,cls='small',anchor='start'):
    parts.append(f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}">{html.escape(value)}</text>')
text(55,40,'MODEL OVERVIEW  /  REPRODUCIBLE LOCAL WORKFLOW','eyebrow')
text(55,94,spec['title'],'title')
text(55,127,spec['subtitle'])
for panel in spec['panels']:
    parts.append(f'<rect x="35" y="{panel["y"]}" width="1730" height="{panel["h"]}" rx="16" fill="#fbfcfe" stroke="#dfe6ee"/>')
    text(60,panel['y']+34,panel['title'],'panel')
for edge in spec['edges']:
    parts.append(f'<path d="{edge["path"]}" class="wire"'+(' stroke-dasharray="7 5"' if edge.get('dashed') else '')+'/>')
    points = re.findall(r'[ML]([0-9.]+),([0-9.]+)', edge['path'])
    (x1,y1),(x2,y2) = [(float(x),float(y)) for x,y in points[-2:]]
    if x2 != x1:
        sign = 1 if x2>x1 else -1
        triangle = f'{x2},{y2} {x2-sign*9},{y2-4} {x2-sign*9},{y2+4}'
    else:
        sign = 1 if y2>y1 else -1
        triangle = f'{x2},{y2} {x2-4},{y2-sign*9} {x2+4},{y2-sign*9}'
    parts.append(f'<polygon points="{triangle}" fill="#536579"/>')
    if edge.get('label'): text(edge['x'],edge['y'],edge['label'])
for box in spec['boxes']:
    x,y,w,h=box['x'],box['y'],box['w'],box['h']
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{box.get("fill", "#edf3fa")}" stroke="{box.get("stroke", "#bacadd")}" stroke-width="1.5"/>')
    for i,line in enumerate(box['lines']):
        text(x+w/2,y+29+i*25,line,'label' if i==0 else 'small','middle')
for item in spec.get('notes',[]): text(item['x'],item['y'],item['text'],item.get('class','small'))
text(55,883,spec['caption'],'foot')
text(55,908,'Original diagram by Yifan Ouyang. Base architecture attribution and training scope: docs/TRAINING.md.','foot')
parts.append('</svg>')
(ROOT/'assets/model-overview.svg').write_text('\n'.join(parts)+'\n')
print('Wrote assets/model-overview.svg')
