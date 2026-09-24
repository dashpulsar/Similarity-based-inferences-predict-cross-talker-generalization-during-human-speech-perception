import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {spawnSync} from 'node:child_process';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';

const root='C:/Users/Alex/Documents/GitHub/cross-talker-generalization-final-review-2026-08-21';
const workspaceDir=path.join(root,'outputs/sbi_parameter_maps_2026-09-18');
const tmp=path.join(workspaceDir,'.build');
const output=path.join(workspaceDir,'presentation');
const skill='C:/Users/Alex/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.11809/skills/presentations';
const python='C:/Users/Alex/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
const {resolvePresentationFont,finalizePresentation}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')));
const font=resolvePresentationFont({fontFamily:'Arial'});
await fs.mkdir(output,{recursive:true});
const p=Presentation.create({slideSize:{width:1280,height:720}});
const sources=[
  ['x21_parameter_landscape','X21: all four conditions','10,969 word responses from 213 participants','x21_all_conditions.html'],
  ['x21_parameter_landscape_no_talker_specific','X21: Talker-specific excluded','8,290 word responses from 161 participants','x21_without_talker_specific.html']
];
for(const [dir,title,subtitle,html] of sources){
  const base=path.join(root,'cross_talker_generalization/analysis/sbi',dir);
  const s=p.slides.add(); s.background.fill='#FFFFFF';
  function text(t,l,y,w,h,size,color='#18232D',bold=false){
    const box=s.shapes.add({geometry:'textbox',position:{left:l,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
    box.text=t;box.text.style={typeface:font,fontSize:size,bold,color,autoFit:'none'};
    return box;
  }
  text(title,56,26,1168,58,44,'#18232D',true);
  text(subtitle,56,84,1168,35,24,'#555F68');
  s.images.add({blob:new Uint8Array(await fs.readFile(path.join(base,'figures/parameter_landscape_3d.png'))),
    contentType:'image/png',fit:'contain',alt:title+'; tau and log10(k) surfaces for signed z and fitted log likelihood',
    position:{left:40,top:120,width:1200,height:530}});
  text('Open interactive map',56,668,500,34,24,'#1768A4');
  text('Training folds 1+2 only',884,668,340,34,24,'#555F68');
  s.speakerNotes.textFrame.setText([
    'Sources: '+path.relative(root,path.join(base,'README.md')),
    'Scores: '+path.relative(root,path.join(base,'landscape.csv')),
    'The grid evaluates 29 tau values and 43 positive k values. Each fit re-estimates the predictor-only GLMM after training-sample standardization.',
    'Left: signed training Wald z. Right: training fitted GLMM log likelihood. Surface connections interpolate visually between fitted grid points.',
    'Tau below one is a non-metric sensitivity. The red point is the expanded-domain grid maximum. Fold 0 is excluded from both fitting and selection.',
    'This is one layer and one training split. Scores across different response samples do not establish comparative predictive performance.',
    'Open '+html+' in a browser to rotate and zoom. Keep both HTML files beside the PPTX. The deck uses external relative hyperlinks and contains static previews.',
  ].join('\n'));
  await fs.copyFile(path.join(base,'figures/parameter_landscape_interactive.html'),path.join(output,html));
}
await(await PresentationFile.exportPptx(p)).save(path.join(tmp,'draft.pptx'));
const patched=spawnSync(python,[path.join(tmp,'add_links.py'),path.join(tmp,'draft.pptx'),path.join(tmp,'linked.pptx')],{encoding:'utf8'});
if(patched.status!==0)throw new Error(patched.stdout+patched.stderr);
const finalPath=path.join(output,'X21_parameter_maps.pptx');
const result=await finalizePresentation({workspaceDir,candidatePath:path.join(tmp,'linked.pptx'),finalPath,
  explicitTotalSlideCount:2,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
  pythonExecutable:python,
  integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit'],
  fontPolicy:{basis:'design',families:[font]},verifyArtifactToolImport:true,
  receiptPath:path.join(tmp,'validation.json')});
console.log(JSON.stringify(result));
const final=await PresentationFile.importPptx(await FileBlob.load(finalPath));
for(let i=0;i<2;i++){
 const slide=final.slides.items[i];
 const png=await final.export({slide,format:'png',scale:1});
 await fs.writeFile(path.join(tmp,`slide-${i+1}.png`),new Uint8Array(await png.arrayBuffer()));
}
console.log(finalPath);
