"""Append an authored deck while preserving existing slide parts verbatim."""
from pathlib import PurePosixPath
import posixpath as pp
import sys
import re
from io import BytesIO
from zipfile import ZipFile, ZIP_DEFLATED
from xml.etree import ElementTree as E

P='http://schemas.openxmlformats.org/presentationml/2006/main'
R='http://schemas.openxmlformats.org/officeDocument/2006/relationships'
REL='http://schemas.openxmlformats.org/package/2006/relationships'
CT='http://schemas.openxmlformats.org/package/2006/content-types'
E.register_namespace('p',P); E.register_namespace('r',R)
def read(path):
    with ZipFile(path) as z: return {n:z.read(n) for n in z.namelist() if not n.endswith('/')}
base, extra=read(sys.argv[1]),read(sys.argv[2])
mapping={n:pp.join(pp.dirname(n),'sbi_added_'+pp.basename(n)) for n in extra
         if n!='[Content_Types].xml' and not n.endswith('.rels')}
assert not set(mapping.values()).intersection(base)
out=dict(base)
for name,new in mapping.items():out[new]=extra[name]
for name,data in extra.items():
    if not name.endswith('.rels') or name=='_rels/.rels':continue
    parent=pp.dirname(pp.dirname(name)); owner=pp.join(parent,pp.basename(name)[:-5])
    newowner=mapping[owner]
    root=E.fromstring(data)
    for rel in root:
        if rel.get('TargetMode')=='External':continue
        target=rel.get('Target')
        absolute=pp.normpath(pp.join(pp.dirname(owner),target)) if not target.startswith('/') else target[1:]
        rel.set('Target',pp.relpath(mapping[absolute],pp.dirname(newowner)))
    out[pp.join(pp.dirname(newowner),'_rels',pp.basename(newowner)+'.rels')]=E.tostring(root,encoding='utf-8',xml_declaration=True)

ct=E.fromstring(base['[Content_Types].xml'])
defaults={x.get('Extension'):x.get('ContentType') for x in ct if x.tag.endswith('Default')}
extra_overrides={x.get('PartName').lstrip('/') for x in E.fromstring(extra['[Content_Types].xml']) if x.tag.endswith('Override')}
for node in E.fromstring(extra['[Content_Types].xml']):
    if node.tag.endswith('Default'):
        ext=node.get('Extension')
        if ext not in defaults:ct.append(node)
        elif defaults[ext]!=node.get('ContentType'):
            for old,new in mapping.items():
                if old.endswith('.'+ext) and old not in extra_overrides:
                    E.SubElement(ct,f'{{{CT}}}Override',PartName='/'+new,ContentType=node.get('ContentType'))
    else:
        old=node.get('PartName').lstrip('/')
        if old in mapping:
            node.set('PartName','/'+mapping[old]);ct.append(node)

pres=E.fromstring(base['ppt/presentation.xml'])
addition=E.fromstring(extra['ppt/presentation.xml'])
assert E.tostring(pres.find(f'{{{P}}}sldSz'))==E.tostring(addition.find(f'{{{P}}}sldSz'))
rels=E.fromstring(base['ppt/_rels/presentation.xml.rels'])
incoming={x.get('Id'):x for x in E.fromstring(extra['ppt/_rels/presentation.xml.rels'])}
used={x.get('Id') for x in rels}
for list_name in ['sldMasterIdLst','notesMasterIdLst','sldIdLst']:
    source=addition.find(f'{{{P}}}{list_name}')
    if source is None:continue
    target=pres.find(f'{{{P}}}{list_name}')
    if target is None:raise ValueError('Expected existing '+list_name)
    ids=[int(x.get('id')) for x in target if x.get('id')]
    for j,node in enumerate(source):
        rel=incoming[node.get(f'{{{R}}}id')]
        rid=f'rIdSBIDiagnostics{list_name}{j}'
        assert rid not in used;used.add(rid)
        old=pp.normpath(pp.join('ppt',rel.get('Target'))).lstrip('/')
        E.SubElement(rels,f'{{{REL}}}Relationship',Id=rid,Type=rel.get('Type'),Target=pp.relpath(mapping[old],'ppt'))
        node.set(f'{{{R}}}id',rid)
        if node.get('id'):
            new_id=max(ids,default=255)+1;ids.append(new_id);node.set('id',str(new_id))
        target.append(node)
out['ppt/presentation.xml']=E.tostring(pres,encoding='utf-8',xml_declaration=True)
out['ppt/_rels/presentation.xml.rels']=E.tostring(rels,encoding='utf-8',xml_declaration=True)
E.register_namespace('',CT)
out['[Content_Types].xml']=E.tostring(ct,encoding='utf-8',xml_declaration=True)
if 'docProps/app.xml' in out:
    app=E.fromstring(out['docProps/app.xml'])
    for node in app.iter():
        if node.tag.endswith('}Slides'):node.text=str(len(pres.find(f'{{{P}}}sldIdLst')))
    out['docProps/app.xml']=E.tostring(app,encoding='utf-8',xml_declaration=True)
allowed={'ppt/presentation.xml','ppt/_rels/presentation.xml.rels','[Content_Types].xml','docProps/app.xml'}
assert all(out[n]==data for n,data in base.items() if n not in allowed)
# PowerPoint may serialize the chart cache at a different decimal precision
# from its workbook. Refresh only numerically identical cache values.
C='http://schemas.openxmlformats.org/drawingml/2006/chart'
S='http://schemas.openxmlformats.org/spreadsheetml/2006/main'
for name in base:
    if '/charts/chart' not in name or not name.endswith('.xml'):continue
    chart=E.fromstring(base[name])
    external=chart.find(f'{{{C}}}externalData')
    if external is None:continue
    relpath=pp.join(pp.dirname(name),'_rels',pp.basename(name)+'.rels')
    target=next(x.get('Target') for x in E.fromstring(base[relpath]) if x.get('Id')==external.get(f'{{{R}}}id'))
    workbook=pp.normpath(pp.join(pp.dirname(name),target))
    with ZipFile(BytesIO(base[workbook])) as z:
        sheets=[n for n in z.namelist() if re.fullmatch(r'xl/worksheets/sheet\d+\.xml',n)]
        assert len(sheets)==1
        cells={c.get('r'):c.find(f'{{{S}}}v').text for c in E.fromstring(z.read(sheets[0])).iter(f'{{{S}}}c') if c.find(f'{{{S}}}v') is not None}
    for ref in chart.iter(f'{{{C}}}numRef'):
        formula=ref.find(f'{{{C}}}f').text
        match=re.search(r'!\$([A-Z]+)\$(\d+):\$([A-Z]+)\$(\d+)$',formula)
        assert match and match[1]==match[3]
        for point in ref.findall(f'{{{C}}}numCache/{{{C}}}pt'):
            value=point.find(f'{{{C}}}v')
            actual=cells[match[1]+str(int(match[2])+int(point.get('idx')))]
            assert abs(float(value.text)-float(actual))<1e-12
            value.text=actual
    out[name]=E.tostring(chart,encoding='utf-8',xml_declaration=True)
with ZipFile(sys.argv[3],'w',ZIP_DEFLATED) as z:
    for n,data in out.items():z.writestr(n,data)
print('Preserved original slide/image/workbook/note/hyperlink parts. Refreshed equivalent numeric chart-cache precision. Total slides:',len(pres.find(f'{{{P}}}sldIdLst')))
