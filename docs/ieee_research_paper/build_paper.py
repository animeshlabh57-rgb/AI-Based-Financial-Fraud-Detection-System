"""Build the IEEE-style PDF and an editable IEEEtran source from one manuscript."""
from pathlib import Path
import csv
import html
import json
import math
import re
import zipfile

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (
    BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, Table, TableStyle,
    KeepTogether, Flowable, BalancedColumns, NextPageTemplate, PageBreak,
)
from reportlab.graphics.shapes import Drawing, Rect, String, Line, PolyLine, Polygon
from reportlab.graphics import renderPDF

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
OUT = ROOT / 'output' / 'pdf'
OUT.mkdir(parents=True, exist_ok=True)
IMAGES = HERE / 'images'
IMAGES.mkdir(exist_ok=True)
DATA = json.loads((HERE / 'manuscript.json').read_text(encoding='utf-8'))
PDF = OUT / 'Financial_Fraud_Detection_IEEE_Paper.pdf'
COL = 252.0

fontdir = Path('C:/Windows/Fonts')
for name, filename in [('TNR', 'times.ttf'), ('TNR-Bold', 'timesbd.ttf'),
                       ('TNR-Italic', 'timesi.ttf'), ('TNR-BoldItalic', 'timesbi.ttf')]:
    pdfmetrics.registerFont(TTFont(name, str(fontdir / filename)))
pdfmetrics.registerFontFamily('TNR', normal='TNR', bold='TNR-Bold',
                              italic='TNR-Italic', boldItalic='TNR-BoldItalic')

STYLES = {
    'body': ParagraphStyle('body', fontName='TNR', fontSize=10, leading=11.3,
        alignment=TA_JUSTIFY, firstLineIndent=10, spaceAfter=3, splitLongWords=True,
        allowWidows=0, allowOrphans=0),
    'abstract': ParagraphStyle('abstract', fontName='TNR-Bold', fontSize=9,
        leading=10.3, alignment=TA_JUSTIFY, spaceAfter=5),
    'keywords': ParagraphStyle('keywords', fontName='TNR-Italic', fontSize=9,
        leading=10.3, alignment=TA_JUSTIFY, spaceAfter=5),
    'h1': ParagraphStyle('h1', fontName='TNR', fontSize=10, leading=12,
        alignment=TA_CENTER, spaceBefore=8, spaceAfter=4, keepWithNext=True),
    'h2': ParagraphStyle('h2', fontName='TNR-Italic', fontSize=10, leading=11.3,
        spaceBefore=5, spaceAfter=3, keepWithNext=True),
    'caption': ParagraphStyle('caption', fontName='TNR', fontSize=8, leading=9.2,
        alignment=TA_CENTER, spaceBefore=4, spaceAfter=4),
    'cell': ParagraphStyle('cell', fontName='TNR', fontSize=8, leading=9.5,
        alignment=TA_CENTER),
    'cellhead': ParagraphStyle('cellhead', fontName='TNR-Bold', fontSize=8,
        leading=9.5, alignment=TA_CENTER),
    'reference': ParagraphStyle('reference', fontName='TNR', fontSize=8, leading=9.2,
        alignment=TA_LEFT, leftIndent=15, firstLineIndent=-15, spaceAfter=3,
        splitLongWords=True),
}


def p(text, style='body'):
    escaped = html.escape(text)
    if style == 'body':
        for old, new in [('b_o(old)', '<i>b</i><sub>o</sub><super>old</super>'),
                         ('b_o', '<i>b</i><sub>o</sub>'), ('b_d', '<i>b</i><sub>d</sub>'),
                         ('q_0.99', '<i>q</i><sub>0.99</sub>'),
                         ('10^(-12)', '10<super>-12</super>')]:
            escaped = escaped.replace(old, new)
    return Paragraph(escaped, STYLES[style])


def arrow(d, x1, y1, x2, y2):
    d.add(Line(x1, y1, x2, y2, strokeWidth=.6))
    a = math.atan2(y2-y1, x2-x1)
    points = [x2, y2, x2-4*math.cos(a-.4), y2-4*math.sin(a-.4),
              x2-4*math.cos(a+.4), y2-4*math.sin(a+.4)]
    d.add(Polygon(points, fillColor=colors.black, strokeWidth=0))


def box(d, x, y, w, h, lines, size=8):
    d.add(Rect(x, y, w, h, strokeColor=colors.black, strokeWidth=.6,
               fillColor=colors.Color(.97, .97, .97)))
    for i, line in enumerate(lines):
        d.add(String(x+w/2, y+h/2+(len(lines)-1)*4.6-i*9.2, line,
                     fontName='TNR', fontSize=size, textAnchor='middle'))


def pipeline():
    d = Drawing(COL, 186)
    box(d, 55, 159, 142, 24, ['15,000 synthetic transactions'])
    arrow(d, 126, 159, 126, 151)
    box(d, 12, 124, 70, 27, ['Train', '9,000 records'])
    box(d, 91, 124, 70, 27, ['Validation', '3,000 records'])
    box(d, 170, 124, 70, 27, ['Test', '3,000 records'])
    d.add(Line(47, 155, 205, 155, strokeWidth=.6))
    arrow(d, 47, 155, 47, 151)
    arrow(d, 126, 155, 126, 151)
    arrow(d, 205, 155, 205, 151)
    arrow(d, 47, 124, 47, 115)
    box(d, 5, 85, 84, 30, ['Fit shared imputation', 'and scaling on train'], 7.7)
    arrow(d, 47, 85, 47, 76)
    box(d, 0, 42, 113, 34, ['SMOTE-NC: 17,730', 'RF / XGBoost / DNN'])
    box(d, 127, 42, 125, 34, ['8,865 legitimate records', 'AE reconstruction training'])
    d.add(Line(47, 81, 189, 81, strokeWidth=.6))
    arrow(d, 189, 81, 189, 76)
    arrow(d, 57, 42, 57, 30)
    arrow(d, 189, 42, 189, 30)
    box(d, 8, 2, 236, 28, ['Validation: checkpoints, thresholds, model ranking',
                             'Test: frozen models and operating points'], 8)
    return d


def precision_recall():
    rows = list(csv.DictReader((ROOT / 'results' / 'test_predictions.csv').open()))
    specs = [('Random Forest', 'RF', .9581, colors.black, None),
             ('XGBoost', 'XGBoost', .9418, colors.Color(.35,.35,.35), [4,2]),
             ('Deep Neural Network', 'DNN', .9307, colors.Color(.1,.3,.65), [1,2]),
             ('Deep Autoencoder', 'AE', .9586, colors.Color(.65,.2,.15), [5,2,1,2])]
    d = Drawing(COL, 188)
    x0, y0, w, h = 33, 28, 211, 111
    for v in [0, .25, .5, .75, 1]:
        y = y0+h*v
        x = x0+w*v
        d.add(Line(x0,y,x0+w,y,strokeColor=colors.Color(.85,.85,.85),strokeWidth=.4))
        d.add(String(x0-5,y-2.5,f'{v:g}',fontName='TNR',fontSize=8,textAnchor='end'))
        d.add(String(x,y0-11,f'{v:g}',fontName='TNR',fontSize=8,textAnchor='middle'))
    d.add(Line(x0,y0,x0,y0+h,strokeWidth=.6))
    d.add(Line(x0,y0,x0+w,y0,strokeWidth=.6))
    d.add(String(x0+w/2,3,'Recall',fontName='TNR',fontSize=9,textAnchor='middle'))
    d.add(String(2, y0+h+6,'Precision',fontName='TNR',fontSize=9))
    for model,label,ap,color,dash in specs:
        values = sorted([(float(r[model+' score']),int(r['isFraud'])) for r in rows],reverse=True)
        total = sum(y for _,y in values)
        tp = 0
        coords = [x0, y0+h]
        i = 0
        while i < len(values):
            score = values[i][0]
            j = i
            while j < len(values) and values[j][0] == score:
                tp += values[j][1]
                j += 1
            coords.extend([x0+w*tp/total,y0+h*tp/j])
            i = j
        d.add(PolyLine(coords, strokeColor=color, strokeWidth=.85, strokeDashArray=dash))
    for index, (_,label,ap,color,dash) in enumerate(specs):
        xx = 18 + (index%2)*123
        yy = 177-(index//2)*13
        d.add(Line(xx,yy,xx+15,yy,strokeColor=color,strokeWidth=1,strokeDashArray=dash))
        d.add(String(xx+20,yy-2.5,f'{label}  AP {ap:.4f}',fontName='TNR',fontSize=8))
    return d


FIGURES = {'pipeline': pipeline(), 'precision_recall': precision_recall()}
for key, drawing in FIGURES.items():
    renderPDF.drawToFile(drawing, str(IMAGES / f'{key}.pdf'))


def roman(n):
    return ['', 'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X'][n]


class Equation(Flowable):
    def __init__(self, text, number):
        Flowable.__init__(self)
        self.text = text
        self.number = number
        self.width = COL
        self.height = 23

    def draw(self):
        formulas = {
            1: 'Δ<sub>o</sub> = b<sub>o</sub><super>new</super> + a - b<sub>o</sub><super>old</super>',
            2: 'Δ<sub>d</sub> = b<sub>d</sub><super>old</super> + a - b<sub>d</sub><super>new</super>.',
            3: 'e(x) = (1/14) ∑<sub>j=1</sub><super>14</super> (z<sub>j</sub> - f<sub>j</sub>(z))<super>2</super>',
            4: 's<sub>AE</sub>(x) = e(x)/(e(x) + q<sub>0.99</sub>).',
            5: 'F<sub>1</sub> = 2TP/(2TP + FP + FN).',
        }
        style = ParagraphStyle('eq',fontName='TNR-Italic',fontSize=10,leading=13,alignment=TA_CENTER)
        paragraph = Paragraph(formulas[self.number], style)
        _, height = paragraph.wrap(COL-22,40)
        paragraph.drawOn(self.canv,0,7)
        self.canv.setFont('TNR', 9)
        self.canv.drawRightString(COL, 8, f'({self.number})')


TITLE_STYLE = ParagraphStyle('title', fontName='TNR',fontSize=22,leading=24,alignment=TA_CENTER)
title_p = Paragraph(html.escape(DATA['title']), TITLE_STYLE)
_, title_height = title_p.wrap(522, 100)
first_top = 738-title_height-93


def first_page(canvas, doc):
    canvas.saveState()
    title_p.drawOn(canvas, 45, 738-title_height)
    author_top = 738-title_height-20
    for index,(name,registration) in enumerate(DATA['authors']):
        xx = 45+index*174+87
        canvas.setFont('TNR', 11)
        canvas.drawCentredString(xx,author_top,name)
        canvas.setFont('TNR',9)
        for row,line in enumerate(DATA['affiliation']):
            canvas.drawCentredString(xx,author_top-14-row*10,line)
        canvas.setFont('TNR',8.5)
        canvas.drawCentredString(xx,author_top-57,'Registration No.: '+registration)
    canvas.restoreState()


def frames(top):
    return [Frame(45,72,COL,top-72,leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0,id='left'),
            Frame(315,72,COL,top-72,leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0,id='right')]


doc = BaseDocTemplate(str(PDF), pagesize=(612,792), leftMargin=45,rightMargin=45,
    topMargin=54,bottomMargin=72, title=DATA['title'],
    author='; '.join(a[0] for a in DATA['authors']),
    subject='IEEE-style conference research paper on a synthetic financial fraud benchmark')
doc.addPageTemplates([PageTemplate(id='first',frames=frames(first_top),onPage=first_page,
                                  autoNextPageTemplate='body'),
                      PageTemplate(id='body',frames=frames(738))])
doc.addPageTemplates([PageTemplate(id='last',frames=[Frame(45,72,522,666,
    leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0,id='balanced')])])

story = [Paragraph('<b>Abstract - </b>'+html.escape(DATA['abstract']), STYLES['abstract']),
         Paragraph('<i>Index Terms - </i>'+html.escape(DATA['keywords'])+'.', STYLES['keywords'])]
section = 0
balance_start = None
for block in DATA['blocks']:
    kind = block['type']
    if kind == 'p' and block['text'].startswith('Reproduction is supported'):
        balance_start = len(story)
    if kind == 'p':
        story.append(p(block['text']))
    elif kind == 'h1':
        section += 1
        story.append(p(roman(section)+'. '+block['text'].upper(),'h1'))
    elif kind == 'h2':
        story.append(p(block['text'],'h2'))
    elif kind == 'equation':
        story.append(Equation(block['plain'],block['number']))
    elif kind == 'figure':
        story.append(KeepTogether([Spacer(1,4),FIGURES[block['id']],p(block['caption'],'caption')]))
    elif kind == 'table':
        cells = [[p(t,'cellhead') for t in block['headers']]]
        cells += [[p(t,'cell') for t in row] for row in block['rows']]
        table = Table(cells,colWidths=[w*COL for w in block['widths']],repeatRows=1,hAlign='CENTER')
        table.setStyle(TableStyle([
            ('LINEABOVE',(0,0),(-1,0),.7,colors.black),
            ('LINEBELOW',(0,0),(-1,0),.4,colors.black),
            ('LINEBELOW',(0,-1),(-1,-1),.7,colors.black),
            ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
            ('TOPPADDING',(0,0),(-1,-1),3),('BOTTOMPADDING',(0,0),(-1,-1),3),
            ('LEFTPADDING',(0,0),(-1,-1),2),('RIGHTPADDING',(0,0),(-1,-1),2),
        ]))
        story.append(KeepTogether([p(block['caption'],'caption'),table,Spacer(1,5)]))
story.append(p('REFERENCES','h1'))
for index,ref in enumerate(DATA['references'],1):
    escaped = html.escape(f'[{index}] '+ref['text'])
    if '[Online]' not in ref['text']:
        escaped = f'<link href="{html.escape(ref["url"],quote=True)}">{escaped}</link>'
    story.append(Paragraph(escaped,STYLES['reference']))
if balance_start is not None:
    final = story[balance_start:]
    story = story[:balance_start]+[NextPageTemplate('last'),PageBreak(),
        BalancedColumns(final,nCols=2,needed=200,leftPadding=0,rightPadding=0,
                        innerPadding=18,topPadding=0,bottomPadding=0)]
doc.build(story)


def tex_escape(text):
    return ''.join({'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$',
                    '#':r'\#','_':r'\_','{':r'\{','}':r'\}',
                    '~':r'\textasciitilde{}','^':r'\textasciicircum{}'}.get(c,c) for c in text)


def tex_text(text):
    pieces = re.split(r'(\[\d+\])',text)
    return ''.join(r'\cite{r'+t[1:-1]+'}' if re.fullmatch(r'\[\d+\]',t) else tex_escape(t) for t in pieces)


tex = [r'\documentclass[conference]{IEEEtran}',
       r'\usepackage{graphicx,amsmath,array,booktabs,url,balance}',
       r'\usepackage[hidelinks]{hyperref}',
       r'\begin{document}',
       r'\title{'+tex_escape(DATA['title'])+'}']
authors = []
for name,registration in DATA['authors']:
    affiliation = r'\\'.join(tex_escape(s) for s in DATA['affiliation'])
    authors.append(r'\IEEEauthorblockN{'+tex_escape(name)+'}\n'+
                   r'\IEEEauthorblockA{'+affiliation+r'\\Registration No.: '+registration+'}')
tex += [r'\author{'+r'\and'.join(authors)+'}',r'\maketitle',
        r'\begin{abstract}',tex_text(DATA['abstract']),r'\end{abstract}',
        r'\begin{IEEEkeywords}',tex_escape(DATA['keywords']),r'\end{IEEEkeywords}']
for block in DATA['blocks']:
    kind = block['type']
    if kind == 'p':
        tex += [tex_text(block['text']), '']
    elif kind == 'h1':
        tex += [r'\section{'+tex_escape(block['text'])+'}']
    elif kind == 'h2':
        tex += [r'\subsection{'+tex_escape(block['text'][3:])+'}']
    elif kind == 'equation':
        tex += [r'\begin{equation}',block['tex'],r'\end{equation}']
    elif kind == 'figure':
        caption = re.sub(r'^Fig\. \d+\. ','',block['caption'])
        tex += [r'\begin{figure}[t]',r'\centering',
                r'\includegraphics[width=\columnwidth]{images/'+block['id']+'.pdf}',
                r'\caption{'+tex_escape(caption)+'}',r'\label{fig:'+block['id']+'}',r'\end{figure}']
    elif kind == 'table':
        caption = re.sub(r'^Table [IVX]+\. ','',block['caption'])
        tex += [r'\begin{table}[t]',r'\caption{'+tex_escape(caption)+'}',r'\centering\footnotesize',
                r'\setlength{\tabcolsep}{3pt}',r'\begin{tabular}{'+'l'+'r'*(len(block['headers'])-1)+'}',
                r'\toprule', ' & '.join(tex_escape(s) for s in block['headers'])+r'\\',r'\midrule']
        tex += [' & '.join(tex_escape(s) for s in row)+r'\\' for row in block['rows']]
        tex += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
tex += [r'\balance',r'\begin{thebibliography}{10}']
for index,ref in enumerate(DATA['references'],1):
    text = ref['text']
    if '[Online]' in text:
        text = text.split(' [Online]. Available:')[0]
    tex += [r'\bibitem{r'+str(index)+'} '+tex_escape(text)+r' \url{'+ref['url']+'}.']
tex += [r'\end{thebibliography}',r'\end{document}']
(HERE/'main.tex').write_text('\n'.join(tex)+'\n',encoding='utf-8')

md = ['# '+DATA['title'],'',
      '; '.join(f'{name} ({registration})' for name,registration in DATA['authors']),
      ', '.join(DATA['affiliation']),'','**Abstract**',DATA['abstract'],'',
      '**Index terms:** '+DATA['keywords'],'']
for block in DATA['blocks']:
    kind=block['type']
    if kind=='p': md += [block['text'],'']
    elif kind=='h1': md += ['## '+block['text'],'']
    elif kind=='h2': md += ['### '+block['text'],'']
    elif kind=='equation': md += ['$$'+block['tex']+r'\tag{'+str(block['number'])+'}$$','']
    elif kind=='figure': md += [block['caption'],'']
    elif kind=='table':
        md += [block['caption'],'','| '+' | '.join(block['headers'])+' |',
               '| '+' | '.join('---' for _ in block['headers'])+' |']
        md += ['| '+' | '.join(row)+' |' for row in block['rows']]+['']
md += ['## References','']+[f'[{i}] {r["text"]} {r["url"]}' for i,r in enumerate(DATA['references'],1)]
(HERE/'manuscript.md').write_text('\n'.join(md)+'\n',encoding='utf-8')

source_zip = HERE/'Financial_Fraud_Detection_IEEE_Source.zip'
with zipfile.ZipFile(source_zip,'w',zipfile.ZIP_DEFLATED) as z:
    for path in [HERE/'main.tex',HERE/'manuscript.md',*IMAGES.glob('*.pdf')]:
        z.write(path,path.relative_to(HERE))
words = sum(len(b.get('text','').split()) for b in DATA['blocks'])+len(DATA['abstract'].split())
print(json.dumps({'pdf':str(PDF),'source_zip':str(source_zip),'main_tex':str(HERE/'main.tex'),
                  'body_words':words,'references':len(DATA['references'])},indent=2))
