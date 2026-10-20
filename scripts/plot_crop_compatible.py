"""Measured recognition figure; installed desktop dependencies only."""
import json,statistics
from pathlib import Path
from reportlab.graphics.shapes import Drawing,String,Rect,Line,Circle
from reportlab.graphics import renderSVG,renderPDF
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import pypdfium2 as pdfium
ROOT=Path(__file__).resolve().parents[1]
pdfmetrics.registerFont(TTFont('Arial',r'C:\Windows\Fonts\arial.ttf'))
pdfmetrics.registerFont(TTFont('ArialBold',r'C:\Windows\Fonts\arialbd.ttf'))
d=json.loads((ROOT/'reports/crop-compatible-results-20261004.json').read_text())
g=json.loads((ROOT/'reports/rare-readout-results-20261004.json').read_text())
p=d['checkpoints']['1000'];canvas=Drawing(1080,740)
canvas.add(Rect(0,0,1080,740,fillColor=HexColor('#ffffff'),strokeColor=None))
def text(x,y,s,size=11,bold=False,color='#263b46'):
 canvas.add(String(x,y,s,fontName='ArialBold' if bold else 'Arial',fontSize=size,fillColor=HexColor(color)))
def line(x1,y1,x2,y2,color='#d9e1e5',width=1):
 canvas.add(Line(x1,y1,x2,y2,strokeColor=HexColor(color),strokeWidth=width))
text(30,699,'Reconnaissance : gains globaux et glyphes rares',23,True)
text(30,671,f"{d['fit_count']} entraînements × 1 000 mises à jour • 5 partitions, 3 graines • DINO et SAM figés",12)
text(30,645,'Planches exclues des adaptations • prototypes fixes • correctif final sans entraînement • aucun modèle activé',11)
text(300,603,'Classe correcte sur crop SAM /254',12,True)
text(795,603,'Rares corrects sur SAM /10',12,True)
text(300,584,'Moyenne et min–max des trois graines',10)
text(795,584,'4 classes ; diagnostic après les fits',10)
x,w,xrare,wrare=300,360,795,205
for val in [0,50,100,150,200,254]:
 xx=x+w*val/254;line(xx,120,xx,563);text(xx-8,105,str(val),9)
for val in [0,2,4,6,8,10]:
 xx=xrare+wrare*val/10;line(xx,120,xx,563);text(xx-5,105,str(val),9)
rows=[('Historique figé','historical',d['historical'],'#7a838c'),('CE de référence','ce',p['ce'],'#315a9b'),('CE équilibrée par classe','balanced',p['balanced'],'#18798a'),('Équilibrage + conservation','preserve',p['preserve'],'#4b7d37'),('Conservation + paires','paired',p['paired'],'#86549c'),('Paires sans conservation KL','paired_no_kl',p['paired_no_kl'],'#b27126'),('Paires + recours historique rare','gated',g['families']['gated'],'#117653')]
assert len(rows)==7 and g['guard']['eligible']
for i,(label,key,e,color) in enumerate(rows):
 y=539-i*58;text(30,y,label,11,True)
 vals=e['masked_correct'];mean=statistics.mean(vals)
 canvas.add(Rect(x,y-7,w*mean/254,22,fillColor=HexColor(color),strokeColor=None))
 lo,hi=min(vals),max(vals);line(x+w*lo/254,y+4,x+w*hi/254,y+4,'#152b37',2)
 for v in vals:canvas.add(Circle(x+w*v/254,y+4,2.3,fillColor=HexColor('#152b37'),strokeColor=None))
 text(x+w*hi/254+10,y,f'{mean:.1f}',11,True)
 text(x,y-23,' / '.join(map(str,vals))+' par graine',9,color='#626d74')
 rare=[g['rare_sam']['gated'][str(seed)]['correct'] for seed in [42,43,44]] if key=='gated' else [d['crop_shift'][key]['seeds'][str(seed)]['rare_sam_correct'] for seed in [42,43,44]]
 rmean=statistics.mean(rare)
 canvas.add(Rect(xrare,y-7,wrare*rmean/10,22,fillColor=HexColor(color),strokeColor=None))
 line(xrare+wrare*min(rare)/10,y+4,xrare+wrare*max(rare)/10,y+4,'#152b37',2)
 for r in rare:canvas.add(Circle(xrare+wrare*r/10,y+4,2.3,fillColor=HexColor('#152b37'),strokeColor=None))
 text(xrare+wrare*max(rare)/10+8,y,f'{rmean:.1f}',10,True)
 text(xrare,y-23,' / '.join(map(str,rare))+' par graine',9,color='#626d74')
text(30,77,'SAM : 254 glyphes appariés sur 343 ; 89 absents ou mal découpés. Scores top-1 sans rejet.',11)
text(30,53,'Dernière ligne : 14 recours historiques, 5 corrections et 1 régression par graine. Règle adaptative sur ces mêmes planches.',10)
text(30,29,'Traits : min–max des graines, sans intervalle de confiance. Corpus déjà examiné ; validation indépendante encore nécessaire.',10,color='#626d74')
renderSVG.drawToFile(canvas,str(ROOT/'reports/crop-compatible-20261004.svg'))
with pdfium.PdfDocument(renderPDF.drawToString(canvas)) as pdf:
 pdf[0].render(scale=2).to_pil().save(ROOT/'reports/crop-compatible-20261004.png')
print(ROOT/'reports/crop-compatible-20261004.png')
