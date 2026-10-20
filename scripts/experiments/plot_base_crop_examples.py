"""Display audited crop pairs and actual predictions, without re-running inference."""
import json
from pathlib import Path
from reportlab.graphics.shapes import Drawing,String,Rect,Line,Image
from reportlab.graphics import renderPDF,renderSVG
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import pypdfium2 as pdfium
ROOT=Path(__file__).resolve().parents[2]
pdfmetrics.registerFont(TTFont('Arial',r'C:\Windows\Fonts\arial.ttf'))
pdfmetrics.registerFont(TTFont('ArialBold',r'C:\Windows\Fonts\arialbd.ttf'))
data=json.loads((ROOT/'reports/base-crop-examples-20261004/examples.json').read_text())
d=Drawing(880,860);d.add(Rect(0,0,880,860,fillColor=HexColor('#ffffff'),strokeColor=None))
def text(x,y,s,size=12,bold=False,color='#263b46'):
 d.add(String(x,y,s,fontName='ArialBold' if bold else 'Arial',fontSize=size,fillColor=HexColor(color)))
text(30,825,'Pourquoi une planche déjà annotée peut encore échouer',21,True)
text(30,798,'Même modèle enrichi après 50 000 mises à jour ; deux crops du même glyphe.',12)
text(115,770,'Crop annoté utilisé à l’entraînement',12,True)
text(485,770,'Crop SAM présenté à l’inférence',12,True)
for i,e in enumerate(data['examples']):
 y=543-i*226
 text(30,y+132,e['label'],12,True)
 for x,key in [(145,'raw_path'),(510,'sam_path')]:
  d.add(Rect(x-1,y-1,182,182,fillColor=HexColor('#ffffff'),strokeColor=HexColor('#e0e6e9')))
  # Both displayed square, as in the classifier's 224x224 input resize.
  d.add(Image(x,y,180,180,str(ROOT/e[key])))
 text(385,y+85,'→',30,True)
 text(145,y-23,'Prédit : '+e['raw_prediction'],12,True,color='#18798a')
 text(510,y-23,'Prédit : '+e['masked_prediction'],12,True,color='#ae423b')
text(30,48,'3 exemples parmi 81 crops appris qui changent de classe après le passage par SAM.',11)
text(30,27,'Planches vues pendant l’entraînement ; diagnostic du pipeline, pas mesure de généralisation.',10,color='#626d74')
renderSVG.drawToFile(d,str(ROOT/'reports/base-crop-examples-20261004.svg'))
with pdfium.PdfDocument(renderPDF.drawToString(d)) as pdf:pdf[0].render(scale=2).to_pil().save(ROOT/'reports/base-crop-examples-20261004.png')
print(ROOT/'reports/base-crop-examples-20261004.png')
