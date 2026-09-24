import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {spawnSync} from 'node:child_process';
import {Presentation,PresentationFile,FileBlob} from '@oai/artifact-tool';
const tmp=path.dirname(fileURLToPath(import.meta.url));
const workspaceDir=path.dirname(tmp);
const root=path.resolve(workspaceDir,'../..');
const skill='C:/Users/Alex/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.11809/skills/presentations';
const python='C:/Users/Alex/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
const {finalizePresentation}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')));
const original=path.join(workspaceDir,'presentation/X21_parameter_maps.pptx');
const backup=path.join(tmp,'X21_parameter_maps_before_sbi_diagnostics.pptx');
try{await fs.copyFile(original,backup,fs.constants.COPYFILE_EXCL);}catch(e){
 if(e.code!=='EEXIST'||!(await fs.readFile(original)).equals(await fs.readFile(backup)))throw e;
}
const p=Presentation.create({slideSize:{width:1280,height:720}});
for(const dataset of ['AN19','X21','B23'])for(const variant of ['base','ft']){
 const rel=`cross_talker_generalization/analysis_update_2026-09-17/si_diagnostics/figures/${dataset}_SBI_${variant}_similarity_all_participants_test_train_ratio.png`;
 const s=p.slides.add();s.background.fill='#FFFFFF';
 s.images.add({blob:new Uint8Array(await fs.readFile(path.join(root,rel))),contentType:'image/png',fit:'contain',
  alt:`${dataset} ${variant}: SBI mean test log loss divided by mean training log loss`,
  position:{left:24,top:24,width:1232,height:672}});
 s.speakerNotes.textFrame.setText(`Source: ${rel}\nEach point shows a participant-held-out fold. Black markers show three-fold means with descriptive 95% fold-bootstrap intervals. The horizontal line marks equal training and test mean log loss. Scores use the same training-fitted model and frozen scaling with random effects set to zero. These plots use the existing fixed-distance predictor. A near-one mean does not establish absence of overfitting.\nInterpretation: cross_talker_generalization/analysis_update_2026-09-18/sbi_diagnostic_review/README.md`);
}
const added=path.join(tmp,'six_sbi_diagnostic_slides.pptx');
await(await PresentationFile.exportPptx(p)).save(added);
const candidatePath=path.join(tmp,'maps_with_sbi_diagnostics_candidate.pptx');
const merge=spawnSync(python,[path.join(tmp,'append_package.py'),backup,added,candidatePath],{encoding:'utf8'});
console.log(merge.stdout);if(merge.status!==0)throw new Error(merge.stderr);
const finalPath=path.join(tmp,'validated/ten_slide_maps_with_diagnostics.pptx');
await fs.mkdir(path.dirname(finalPath),{recursive:true});
const result=await finalizePresentation({workspaceDir,candidatePath,finalPath,
 explicitTotalSlideCount:10,requiredNativeChartOwnerSlides:[3,4],requiredNativeTableOwnerSlides:[],
 pythonExecutable:python,integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit'],
 verifyArtifactToolImport:true,receiptPath:path.join(tmp,'diagnostics_append_validation.json')});
console.log(JSON.stringify({finalPath:result.finalPath,slides:10}));
const finished=await PresentationFile.importPptx(await FileBlob.load(finalPath));
for(let i=0;i<finished.slides.items.length;i++){
 const png=await finished.export({slide:finished.slides.items[i],format:'png',scale:1});
 await fs.writeFile(path.join(tmp,`after-diagnostics-${i+1}.png`),new Uint8Array(await png.arrayBuffer()));
}
