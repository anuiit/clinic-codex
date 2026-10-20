#!/usr/bin/env python3
"""Standalone measured chart, rendered with the bundled desktop ReportLab."""
import json,statistics
from pathlib import Path
from reportlab.graphics.shapes import Drawing,String,Rect
from reportlab.graphics.charts.barcharts import HorizontalBarChart
from reportlab.graphics import renderSVG
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
ROOT=Path(__file__).resolve().parents[2]
pdfmetrics.registerFont(TTFont('Arial',r'C:\Windows\Fonts\arial.ttf'))
pdfmetrics.registerFont(TTFont('ArialBold',r'C:\Windows\Fonts\arialbd.ttf'))
d=json.loads((ROOT/'reports/recognition-mechanisms-results-20261003.json').read_text())
drawing=Drawing(960,620)
drawing.add(Rect(0,0,960,620,fillColor=HexColor('#ffffff'),strokeColor=None))
def text(x,y,value,size=11,bold=False,color='#263b46'):
    drawing.add(String(x,y,value,fontName='ArialBold' if bold else 'Arial',fontSize=size,fillColor=HexColor(color)))
def bars(names,values,y,height):
    chart=HorizontalBarChart();chart.x=255;chart.y=y;chart.width=590;chart.height=height
    chart.data=[values[::-1]];chart.categoryAxis.categoryNames=names[::-1]
    chart.valueAxis.valueMin=0;chart.valueAxis.valueMax=200;chart.valueAxis.valueStep=50
    chart.categoryAxis.labels.fontName='Arial';chart.categoryAxis.labels.fontSize=11
    chart.valueAxis.labels.fontName='Arial';chart.valueAxis.labels.fontSize=10
    chart.bars[0].fillColor=HexColor('#18798a');chart.bars[0].strokeColor=None;chart.barLabelFormat='%0.1f'
    chart.barLabels.fontName='ArialBold';chart.barLabels.fontSize=11
    chart.barLabels.nudge=8;chart.barLabels.boxAnchor='w'
    drawing.add(chart)
text(30,583,'Glyphes : résultats des expériences contrôlées',23,True)
text(30,560,'3 octobre 2026 • 120 entraînements courts • aucun candidat activé',12)
text(30,525,'Même boîte SAM, pixels du crop différents',15,True)
names=['Masque blanc','Rectangle brut','Masque dilaté de 3 px','Fond blanchi à 50 %']
values=[d['crops'][k]['splits']['train']['top1_correct'] for k in ['mask','rectangle','dilated3','half_background']]
assert values==[160,125,127,116]
bars(names,values,365,140)
text(255,336,'Noms corrects /254 propositions appariées — classifieur historique figé',10)
text(30,300,'Apprentissage : classification seule et ajout de contraste',15,True)
records=[d['contrast']['historical']]+[d['contrast'][k] for k in ['ce_fixed','ce_crosspage_01','ce_crosspage_05']]+[d['paired_batches'][k] for k in ['ce_fixed','ce_crosspage_01']]
names=['Historique :160','CE :170–172','CE +contraste λ=0,1 :168–170','CE +contraste λ=0,5 :165–168','CE, paires imposées :169–172','CE +contraste, paires :169–173']
values=[statistics.mean(r['masked_correct']) for r in records]
bars(names,values,100,180)
text(255,72,'Barres : moyenne des 3 seeds ; plages indiquées à gauche : min–max, pas des IC',10)
text(30,42,'17 planches retirées tour à tour • 5 folds • 343 glyphes connus, dont 254 appariés par SAM',11)
text(30,22,'Le corpus a déjà été examiné : ces résultats ne constituent pas une validation indépendante.',10,color='#626d74')
p=ROOT/'reports/recognition-mechanisms-20261003.svg';renderSVG.drawToFile(drawing,str(p))
from reportlab.graphics import renderPDF
import pypdfium2 as pdfium
with pdfium.PdfDocument(renderPDF.drawToString(drawing)) as pdf:
    image=pdf[0].render(scale=2).to_pil()
    image.save(ROOT/'reports/recognition-mechanisms-20261003.png')
print(p)


