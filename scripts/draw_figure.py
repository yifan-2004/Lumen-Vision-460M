#!/usr/bin/env python3
"""Draw the scientific architecture plate from the implemented tensor operations."""
from pathlib import Path
import base64
import io
import math
import html
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'assets/model-overview.svg'
S=['<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="2200" height="1320" viewBox="0 0 2200 1320">',
'<title>Lumen Vision 460M: tensor compression, multimodal decoder and answer-supervised connector adaptation</title>',
'<rect width="2200" height="1320" fill="white"/>',
'<style>text{font-family:Arial,Helvetica,sans-serif;fill:#1f2939}.panel{font-size:26px;font-weight:700}.label{font-size:21px;font-weight:600}.body{font-size:18px}.small{font-size:16px;fill:#556477}.math{font-family:Georgia,Times New Roman,serif;font-style:italic;font-size:24px}.tiny{font-size:14px;fill:#667388}</style>']
C={'ink':'#384659','teal':'#197c8b','lite':'#e6f3f4','purple':'#86549d','plite':'#f2eaf6','amber':'#b98934','al':'#fbf1dd','gray':'#e9edf2'}

def txt(x,y,t,cl='body',anchor='start',fill=None):
 S.append(f'<text x="{x}" y="{y}" class="{cl}" text-anchor="{anchor}"'+(f' style="fill:{fill}"' if fill else '')+'>'+html.escape(t)+'</text>')
def rect(x,y,w,h,fill='white',stroke='#c1ccd6',r=3,sw=1.4):
 S.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
def line(x1,y1,x2,y2,color=C['ink'],sw=1.7,dash=False):
 S.append(f'<path d="M{x1},{y1} L{x2},{y2}" fill="none" stroke="{color}" stroke-width="{sw}"'+(' stroke-dasharray="6 5"' if dash else '')+'/>')
def path(points,color=C['ink'],sw=1.8,dash=False,arrow=True):
 s='M'+ ' L'.join(f'{x},{y}' for x,y in points)
 S.append(f'<path d="{s}" fill="none" stroke="{color}" stroke-width="{sw}"'+(' stroke-dasharray="7 5"' if dash else '')+'/>')
 if arrow:
  (x1,y1),(x,y)=points[-2:];a=math.atan2(y-y1,x-x1);n=10;spread=.42
  pts=[(x,y),(x-n*math.cos(a-spread),y-n*math.sin(a-spread)),(x-n*math.cos(a+spread),y-n*math.sin(a+spread))]
  S.append('<polygon points="'+' '.join(f'{a},{b}' for a,b in pts)+f'" fill="{color}"/>')
def arrow(x1,y1,x2,y2,**kw):path([(x1,y1),(x2,y2)],**kw)
def labelbox(x,y,w,h,text,fill=C['lite'],stroke=C['teal'],font='body'):
 rect(x,y,w,h,fill,stroke);txt(x+w/2,y+h/2+6,text,font,'middle')
def circle(x,y,label,color=C['ink'],r=13):
 S.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="white" stroke="{color}" stroke-width="1.6"/>');txt(x,y+6,label,'body','middle')
def grid(x,y,n,size,fill,stroke='#91b7bf',highlight=False):
 rect(x,y,size,size,fill,stroke,0)
 for j in range(1,n):line(x+j*size/n,y,x+j*size/n,y+size,stroke,.65);line(x,y+j*size/n,x+size,y+j*size/n,stroke,.65)
 if highlight:rect(x+size*.375,y+size*.375,size*.125,size*.125,C['plite'],C['purple'],0,2.6)
def cube(x,y,size,n,depth=18,fill=C['lite'],highlight=False):
 S.append(f'<path d="M{x},{y} l{depth},-{depth} h{size} v{size} l-{depth},{depth} Z" fill="#d6e7eb" stroke="#8eabb3"/>')
 grid(x,y,n,size,fill,highlight=highlight)
def badge(x,y,t,train=False):
 color=C['purple'] if train else '#61768a';txt(x,y,t,'small',fill=color)
 if not train:
  rect(x-19,y-12,11,10,'#ecf0f3',color,2,1);S.append(f'<path d="M{x-17},{y-12} v-4 a3.5,3.5 0 0 1 7,0 v4" fill="none" stroke="{color}"/>')

# Publication panels: no UI cards; architecture detail and explicit tensor algebra.
txt(55,48,'(a) Visual encoding and spatial-to-channel compression','panel')
txt(1160,48,'(b) Multimodal sequence and causal decoder','panel')
line(1105,30,1105,790,'#d2d9e1',1)
line(55,790,2145,790,'#d2d9e1',1)

image=Image.open(ROOT/'hf-inference/data/img.jpg').convert('RGB');image.thumbnail((320,240));buf=io.BytesIO();image.save(buf,format='JPEG',quality=86)
uri='data:image/jpeg;base64,'+base64.b64encode(buf.getvalue()).decode()
S.append(f'<image x="65" y="105" width="215" height="175" preserveAspectRatio="xMidYMid slice" xlink:href="{uri}"/>')
rect(65,105,215,175,'none','#8494a6',0)
# Four sample image crops express tiling, with an explicit schematic label.
for j in range(2):
 for i in range(2):
  x,y=335+i*89,107+j*89
  S.append(f'<svg x="{x}" y="{y}" width="80" height="80" viewBox="{i*110} {j*90} 110 90"><image width="220" height="180" xlink:href="{uri}"/></svg>')
  rect(x,y,80,80,'none','#8599ab',0)
txt(172,310,'Image I','math','middle');txt(418,310,'Global + local tiles','body','middle')
arrow(290,190,325,190);arrow(520,190,563,190)
for k in reversed(range(4)):
 rect(590+k*8,119-k*8,155,155,'#edf3f8','#8fa8be',3)
labelbox(602,139,130,37,'Patch embed', '#f5f8fb','#9bb1c5')
labelbox(602,190,130,52,'ViT block', '#e9f0f7','#9bb1c5')
txt(726,255,'×12','small','end');badge(615,285,'frozen encoder')
arrow(763,190,805,190);cube(825,123,145,32,fill='#e7f3f4',highlight=True)
txt(897,310,'32 × 32 × 768','math','middle')
txt(65,352,'512 × 512 tile • patch size 16 • one dense feature map per tile','small')
path([(897,326),(897,368),(165,368),(165,411)],'#8093a5',1.5,dash=True)

# Actual pixel-shuffle: 4x4 spatial neighborhood becomes 16 channel groups.
txt(65,407,'4 × 4 neighborhood','label')
cube(77,442,176,32,fill='#e7f3f4',highlight=True)
palette=['#507e9b','#658ca3','#7b9eae','#8eafba','#70a295','#81af9f','#91bda9','#a2c9b7','#b19e72','#bda97d','#c9b98e','#d6c6a0','#a284ad','#b092b8','#bea3c4','#cbb3d0']
# Zoomed 16 positions make the permutation visible, not a generic arrow.
for j in range(4):
 for i in range(4):rect(292+i*26,456+j*26,23,23,palette[j*4+i],'white',0,.8)
path([(165,510),(275,510)],C['purple'])
txt(344,594,'16 spatial vectors','small','middle')
arrow(409,510,453,510,color=C['purple'])
for i,color in enumerate(palette):rect(464+i*6,440,6,150,color,'white',0,.35)
txt(513,619,'16 × 768','math','middle');txt(513,646,'channel concatenation','small','middle')
arrow(574,510,613,510,color=C['purple'])
for i in range(10):
 for j in range(6):rect(633+j*13,451+i*12,12,11,'#cdb5db' if (i+j)%3 else '#a27db6','white',0,.3)
txt(671,428,'Wᵀ','math','middle',C['purple']);txt(672,598,'12288 → 960','small','middle');badge(627,627,'trainable',True)
arrow(722,510,767,510,color=C['purple']);cube(789,442,152,8,fill='#e9f4f4')
txt(866,619,'8 × 8 × 960','math','middle');txt(866,646,'64 visual tokens / tile','small','middle')
path([(960,510),(1044,510),(1044,150),(1150,150)],C['teal'],2.0)
txt(65,700,'Z ∈ ℝ¹⁰²⁴×⁷⁶⁸  →  reshape / permute  →  ℝ⁶⁴×¹²²⁸⁸  →  Zᵥ ∈ ℝ⁶⁴×⁹⁶⁰','math')
txt(65,741,'Total visual tokens scale with the number of tiles; 64 is not the total for every image.','small')

# Fusion is token substitution, not cross attention.
txt(1160,99,'Visual embeddings Zᵥ','label',fill=C['teal'])
for i in range(9):
 for j in range(5):rect(1160+i*40,120+j*11,33,9,['#97c8cf','#6aabb8','#b6dadf'][ (i+j)%3],'none',0)
txt(1540,151,'…','label');txt(1580,151,'Nᵥ × 960','math')
txt(1160,211,'Prompt token embeddings E(q)','label',fill=C['amber'])
for i,w in enumerate(['Describe','this','image','.']):
 labelbox(1160+i*105,231,96,34,w,C['al'],'#d7be8d','small')
path([(1513,177),(1710,177),(1710,290),(1470,290),(1470,329)],C['teal']);path([(1580,248),(1810,248),(1810,329)],C['amber'])
txt(1160,310,'[ image tokens ; text tokens ]','math')
for i in range(18):rect(1160+i*43,329,36,43,C['lite'] if i<12 else C['al'],'#87b8c0' if i<12 else '#ceb687',2)
txt(1160,397,'Visual token replacement → one causal attention stream','small')
path([(1160,350),(1140,350),(1140,474),(1304,474),(1304,488)])
# Repeated decoder shell, with real residual attention and SwiGLU operations.
for dx in [12,6,0]:rect(1160+dx,430-dx,790,310,'#fafbfd' if dx else '#f8fafc','#bbc6d3',4)
txt(1175,461,'Decoder block','label');txt(1932,461,'×32','label','end')
badge(1787,490,'frozen parameters')
labelbox(1184,488,240,32,'RMSNorm','#eef1f5','#a3b1c1','small')
labelbox(1184,534,240,97,'RoPE + grouped QKV', '#e6eef7','#8da6bd','body')
txt(1304,610,'15 Q heads / 5 KV heads','small','middle')
arrow(1304,520,1304,532);arrow(1304,631,1304,660)
circle(1304,675,'+');path([(1184,504),(1170,504),(1170,675),(1288,675)],'#708196',1.5)
path([(1304,690),(1304,709),(1514,709),(1514,506),(1530,506)])
labelbox(1530,488,190,32,'RMSNorm','#eef1f5','#a3b1c1','small')
# Split gates and up projection, elementwise product and down projection.
labelbox(1470,549,110,35,'Wgate',C['plite'],'#baa1cb','small')
labelbox(1630,549,110,35,'Wup',C['plite'],'#baa1cb','small')
path([(1625,520),(1625,537),(1525,537),(1525,548)]);path([(1625,537),(1685,537),(1685,548)])
labelbox(1470,603,110,30,'SiLU',C['plite'],'#baa1cb','small');arrow(1525,584,1525,602)
circle(1625,648,'×',C['purple']);path([(1525,633),(1525,648),(1610,648)],C['purple']);path([(1685,584),(1685,648),(1640,648)],C['purple'])
labelbox(1772,628,134,37,'Wdown',C['plite'],'#baa1cb','small');arrow(1640,648,1770,648,color=C['purple'])
circle(1915,686,'+');path([(1839,665),(1839,686),(1900,686)])
path([(1514,506),(1514,474),(1940,474),(1940,686),(1930,686)],'#708196',1.5)
txt(1609,690,'SwiGLU','small','middle')
path([(1915,702),(1915,754),(2040,754),(2040,687)])
labelbox(1980,600,165,80,'Norm + head','#edf2f6','#a6b5c6','body')
arrow(2062,599,2062,538)
for i,h in enumerate([12,28,45,62,30,15]):rect(1995+i*23,520-h,14,h,'#5f97a3','none',1)
txt(2062,445,'p(yₜ | I, q, y₍<t₎)','math','middle')
labelbox(1950,225,195,116,'A polar bear …','#edf6ef','#a6c7ad')
txt(2047,365,'Generated answer','small','middle')
path([(2135,490),(2160,490),(2160,280),(2145,280)])

# Training illustration: target tensor, mask and gradient through frozen network.
txt(55,854,'(c) Answer-only supervision and connector gradients','panel')
txt(55,900,'Full token stream','body');txt(55,959,'Next-token labels','body');txt(55,1018,'Loss mask mₜ','body')
words=['<image>','<image>','…','question','assistant','A','polar','bear','<eos>']
for i,w in enumerate(words):
 x=230+i*106;col=C['lite'] if i<3 else C['al'] if i<5 else '#e5f1e8'
 labelbox(x,878,96,36,w,col,'#bec9cf','small')
 labelbox(x,937,96,36,'−100' if i<4 else ['A','polar','bear','<eos>','—'][i-4], '#eef0f3' if i<4 else '#e5f1e8','#c4cbd2','small')
 labelbox(x,996,96,32,'0' if i<4 or i==8 else '1','#eef0f3' if i<4 or i==8 else '#d4e8da','#c4cbd2','small')
txt(230,1068,'Prompt tokens are ignored; targets are shifted once before the original forward call.','small')
# Correct visual alignment: last input has no next-token target and is excluded from x.
txt(230,1100,'The full answer + EOS is appended; training input excludes the last token (schematic).','tiny')
txt(1260,915,'L = − (1/N) ∑ₜ mₜ log p(yₜ | I, q, y₍<t₎)','math')
txt(1260,939,'N = ∑ₜ mₜ','small')
txt(1260,981,'θvision, θdecoder fixed; only W is updated.','math')
txt(1260,1021,'W ← AdamW(W, ∂L/∂W)','math',fill=C['purple'])
path([(1560,1037),(1560,1145),(28,1145),(28,670),(671,670),(671,641)],C['purple'],2.2,dash=True)
rect(743,1127,770,36,'white','none')
txt(756,1153,'Gradient propagates through the frozen decoder to the visual connector.','body',fill=C['purple'])
txt(55,1210,'Blue / teal: visual tensors and backbone   •   Gold: text tokens   •   Purple: connector parameters / gradient   •   Green: answer targets','small')
line(55,1240,2145,1240,'#cbd4df',1)
txt(55,1272,'Lumen Vision 460M — tensor operations follow the implemented SigLIP2 / SmolLM2 base; schematic feature values, not measured attention.','small')
txt(55,1300,'Original architecture: Tether AI Research / VisionPsy Nano. Redrawn scientific illustration and connector adaptation tooling: Yifan Ouyang.','tiny')
S.append('</svg>');OUT.write_text('\n'.join(S)+'\n');print(OUT)
