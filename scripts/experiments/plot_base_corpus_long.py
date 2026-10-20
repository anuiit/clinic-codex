"""Static measured base-training chart using already bundled ReportLab."""
import json
from pathlib import Path
from reportlab.graphics.shapes import Drawing,String,Rect,Line,PolyLine,Circle
from reportlab.graphics import renderSVG,renderPDF
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import pypdfium2 as pdfium
ROOT=Path(__file__).resolve().parents[2]
pdfmetrics.registerFont(TTFont('Arial',r'C:\Windows\Fonts\arial.ttf'))
pdfmetrics.registerFont(TTFont('ArialBold',r'C:\Windows\Fonts\arialbd.ttf'))
d=json.loads((ROOT/'reports/base-corpus-long-results-20261004.json').read_text())
views=json.loads((ROOT/'reports/base-inference-views-results-20261004.json').read_text())
canvas=Drawing(1000,710);canvas.add(Rect(0,0,1000,710,fillColor=HexColor('#ffffff'),strokeColor=None))
def text(x,y,s,size=11,bold=False,color='#263b46'):
    canvas.add(String(x,y,s,fontName='ArialBold' if bold else 'Arial',fontSize=size,fillColor=HexColor(color)))
def line(x1,y1,x2,y2,color='#d9e1e5',width=1):canvas.add(Line(x1,y1,x2,y2,strokeColor=HexColor(color),strokeWidth=width))
text(30,677,'Ajout des annotations : départ des poids historiques',22,True)
text(30,652,'11 entraînements distincts × 50 000 mises à jour • DINO et SAM figés • aucun modèle activé',12)
text(30,617,'Mesure principale : 17 planches exclues à tour de rôle, 5 partitions, graine 42',13,True)
text(30,595,'Bonne boîte SAM + bonne classe, moyenne par planche (%)',11)
x0,y0,width,height=105,345,820,220
for v in range(0,81,20):
    y=y0+height*v/80;line(x0,y,x0+width,y);text(x0-35,y-4,str(v),10)
for v in range(0,51,10):
    x=x0+width*v/50;line(x,y0,x,y0-5,color='#546974');text(x-8,y0-20,str(v),10)
line(x0,y0,x0+width,y0,color='#546974');line(x0,y0,x0,y0+height,color='#546974')
text(330,305,'Mises à jour de la tête de reconnaissance (milliers)',11)
colors={'parent':'#315a9b','combined':'#18798a'}
for arm in ['parent','combined']:
    points=[]
    for epoch,e in sorted(d['curves'][arm].items(),key=lambda x:int(x[0])):
        points.extend([x0+width*int(epoch)/100,y0+height*100*e['page_mean_joint']/80])
    canvas.add(PolyLine(points,strokeColor=HexColor(colors[arm]),strokeWidth=2.7))
base=d['historical']['page_mean_joint']*100;y=y0+height*base/80
canvas.add(Line(x0,y,x0+width,y,strokeColor=HexColor('#7a838c'),strokeWidth=1.3,strokeDashArray=[5,4]))
for x,arm,name in [(105,'parent','Corpus ancien seul'),(370,'combined','Corpus ancien + annotations')]:
    line(x,580,x+25,580,colors[arm],3);text(x+33,576,name,11)
line(695,580,720,580,'#7a838c',1.5);text(728,576,'Classifieur historique',11)
rectangle=views['families']['warm_combined']['rectangle']
yr=y0+height*100*rectangle['page_mean_joint']/80
canvas.add(Circle(x0+width,yr,4,fillColor=HexColor('#996b18'),strokeColor=None))
text(700,yr+12,f"Rectangle final : {100*rectangle['page_mean_joint']:.1f} %",10,True,color='#996b18')
text(30,272,'Résultat final, fixé à 100 époques',14,True)
text(325,248,'Boîtes métier : classe correcte /343',11,True)
text(655,248,'Boîtes SAM : boîte + classe /343',11,True)
rows=[('Classifieur historique',d['historical'],'#7a838c'),('Recette longue, corpus ancien',d['primary']['parent'],colors['parent']),('Recette longue, corpus enrichi',d['primary']['combined'],colors['combined']),('Enrichi, inférence rectangle',dict(raw=d['primary']['combined']['raw'],masked_correct=rectangle['correct']),'#996b18')]
for row,(name,e,color) in enumerate(rows):
    y=216-row*44;text(30,y-1,name,11)
    for x,value in [(325,e['raw']['correct']),(655,e['masked_correct'])]:
        canvas.add(Rect(x,y-7,260*value/343,20,fillColor=HexColor(color),strokeColor=None))
        text(x+260*value/343+8,y-1,f'{value}/343 ({100*value/343:.1f} %)',10,True)
text(30,58,'SAM : 254 glyphes appariés sur 343 ; les 89 absents ou mal découpés comptent comme des échecs.',11)
text(30,37,'Courbe : moyenne par planche. Barres : total des glyphes. Le meilleur-dev reste secondaire.',10,color='#626d74')
text(30,17,'Corpus déjà examiné et provenance initiale inconnue : validation de l’adaptation, pas test aveugle.',10,color='#626d74')
renderSVG.drawToFile(canvas,str(ROOT/'reports/base-corpus-long-20261004.svg'))
with pdfium.PdfDocument(renderPDF.drawToString(canvas)) as pdf:pdf[0].render(scale=2).to_pil().save(ROOT/'reports/base-corpus-long-20261004.png')
print(ROOT/'reports/base-corpus-long-20261004.png')
