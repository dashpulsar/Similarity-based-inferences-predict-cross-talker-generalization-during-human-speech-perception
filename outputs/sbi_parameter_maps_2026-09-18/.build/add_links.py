"""Add relative HTML hyperlinks to the generated slide previews and link labels."""
import sys
from zipfile import ZipFile, ZIP_DEFLATED
from xml.etree import ElementTree as ET

P='http://schemas.openxmlformats.org/presentationml/2006/main'
A='http://schemas.openxmlformats.org/drawingml/2006/main'
R='http://schemas.openxmlformats.org/officeDocument/2006/relationships'
REL='http://schemas.openxmlformats.org/package/2006/relationships'
ET.register_namespace('p',P); ET.register_namespace('a',A); ET.register_namespace('r',R)
with ZipFile(sys.argv[1]) as src:
    files={n:src.read(n) for n in src.namelist()}
for i,target in enumerate(['x21_all_conditions.html','x21_without_talker_specific.html'],1):
    name=f'ppt/slides/slide{i}.xml'
    relname=f'ppt/slides/_rels/slide{i}.xml.rels'
    slide=ET.fromstring(files[name]); relationships=ET.fromstring(files[relname])
    rid='rIdInteractiveHTML'
    ET.SubElement(relationships,f'{{{REL}}}Relationship',Id=rid,Type=R+'/hyperlink',Target=target,TargetMode='External')
    linked=0
    for shape in slide.iter(f'{{{P}}}sp'):
        if ''.join(t.text or '' for t in shape.iter(f'{{{A}}}t'))=='Open interactive map':
            for run in shape.iter(f'{{{A}}}r'):
                props=run.find(f'{{{A}}}rPr')
                if props is None:
                    props=ET.Element(f'{{{A}}}rPr');run.insert(0,props)
                props.set('u','sng')
                ET.SubElement(props,f'{{{A}}}hlinkClick',{f'{{{R}}}id':rid,'tooltip':'Open rotatable map in a browser'})
            linked+=1
    assert linked==1
    for picture in slide.iter(f'{{{P}}}pic'):
        props=picture.find(f'{{{P}}}nvPicPr/{{{P}}}cNvPr')
        ET.SubElement(props,f'{{{A}}}hlinkClick',{f'{{{R}}}id':rid,'tooltip':'Open rotatable map in a browser'})
    files[name]=ET.tostring(slide,encoding='utf-8',xml_declaration=True)
    files[relname]=ET.tostring(relationships,encoding='utf-8',xml_declaration=True)
with ZipFile(sys.argv[2],'w',ZIP_DEFLATED) as dest:
    for n,b in files.items():dest.writestr(n,b)
