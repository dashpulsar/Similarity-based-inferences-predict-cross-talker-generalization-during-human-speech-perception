import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';

const tmp=path.dirname(fileURLToPath(import.meta.url));
const workspaceDir=path.dirname(tmp);
const skill='C:/Users/Alex/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.11809/skills/presentations';
const python='C:/Users/Alex/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
const {resolvePresentationFont,applyPresentationChartFont,finalizePresentation}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')));
const font=resolvePresentationFont({fontFamily:'Arial'});
const evidence=JSON.parse(await fs.readFile(path.join(tmp,'evidence.json'),'utf8'));
const p=Presentation.create({slideSize:{width:1280,height:720}});
const colors=['#727A83','#CC7837','#286499','#198575'];
function text(slide,value,left,top,width,height,size=25,color='#263340',bold=false){
  const shape=slide.shapes.add({geometry:'textbox',position:{left,top,width,height},fill:'none',line:{fill:'none',width:0}});
  shape.text=value;
  shape.text.style={typeface:font,fontSize:size,color,bold,autoFit:'none'};
  return shape;
}
function chart(slide,categories,values,max,unit,format){
  const ch=slide.charts.add('bar',{
    position:{left:72,top:154,width:1136,height:398},categories,
    series:evidence.labels.map((name,i)=>({name,values:values.map(v=>Number(v[i].toFixed(6))),fill:colors[i],valuesFormatCode:format})),
    barOptions:{direction:'column',grouping:'clustered',gapWidth:110,overlap:0},
    hasLegend:true,legend:{position:'bottom',overlay:false,textStyle:{typeface:font,fontSize:20}},
    xAxis:{textStyle:{typeface:font,fontSize:23},majorGridlines:null},
    yAxis:{min:0,max,majorUnit:unit,title:{text:format==='0.00'?'Pearson r with listener accuracy':'Mean Wald z',textStyle:{typeface:font,fontSize:21}},numberFormatCode:format==='0.00'?'0.0':'0',textStyle:{typeface:font,fontSize:19},majorGridlines:{fill:'#E3E6E9',width:1}},
    dataLabels:{showValue:true,position:'outEnd',textStyle:{typeface:font,fontSize:21}},
    chartFill:'#FFFFFF',plotAreaFill:'#FFFFFF',chartLine:{fill:'none',width:0}
  });
  applyPresentationChartFont(ch,{fontFamily:font});
  return ch;
}
let s=p.slides.add();s.background.fill='#FFFFFF';
text(s,'Acoustic baselines rival HuBERT in AN19',56,30,1168,64,44,'#172E45',true);
text(s,'Historical test-fold refits: three-fold mean z, with HuBERT shown at Tr-24',56,104,1168,36,25,'#53616E');
chart(s,['AN19','X21'],[evidence.historical_z.AN19,evidence.historical_z.X21],16,4,'0.0');
text(s,'AN19 STRF also matches the best base layer: 13.94 vs 13.69 (Tr-12).',66,570,1150,36,27,'#172E45',true);
text(s,'X21 shows a clearer HuBERT advantage. The cause of this difference remains open.',66,614,1150,38,25);
text(s,'Descriptive rankings. A paired test of representation differences is still needed.',66,670,1150,29,20,'#687582');
s.speakerNotes.textFrame.setText([
  'I am focusing on the relative ranking. In AN19, STRF is slightly above even the best historical base-layer mean, while X21 favors HuBERT. These are averages of historical target-refit z values, not fixed-model held-out prediction scores. The plotted neural models both use Tr-24, so there is no best-layer selection in the bars. The annotation separately identifies the highest base-layer mean.',
  'The same positive ceiling rescales all representations within a dataset, so ceiling normalization cannot cause the within-dataset ranking. No paired significance test supports a claim that STRF is superior. The current analysis changes several conventions: it uses negative training-standardized distance at tau=2, fits on two participant folds and scores the remaining fold. Its AN19 predictor-only training z is 4.014 for base Tr-24 and 3.817 for STRF. Thus the exact ranking is analysis-dependent.',
  'Source: cross_talker_generalization/analysis/model_comparison/reference_checks/z_value_review/tables/historical_sbi_fold_z.csv. Historical mean z: AN19 MFCC 11.720367, STRF 13.942255, base Tr-24 10.825501, FT Tr-24 9.028088. X21: 5.736440, 5.723491, 7.761770, 7.679416. Source for current scores: artifacts/models/AN19-AN19_acoustic-confirmatory-full-20260906/coefficients.csv and AN19-AN19_hubert_base_tsne-confirmatory-tr24-20260906/coefficients.csv.',
].join('\n\n'));

s=p.slides.add();s.background.fill='#FFFFFF';
const rows=evidence.descriptive.filter(x=>x.dataset==='AN19');
text(s,'A clue in AN19: differences between words',56,30,1168,64,44,'#172E45',true);
text(s,'Current fixed-distance predictors: correlation with mean listener accuracy',56,104,1168,36,25,'#53616E');
chart(s,['Across all cells','After removing word means'],[rows.map(r=>r.r_raw),rows.map(r=>r.r_word)],0.6,0.2,'0.00');
text(s,'After word centering, HuBERT has the stronger association.',66,570,1150,38,28,'#172E45',true);
text(s,'Word-dependent variation is a plausible contributor. Its acoustic source remains unresolved.',66,615,1150,38,24);
text(s,'574 condition × recording cells, 48 words. Equal cell weights. Descriptive r, without inference.',66,669,1150,31,20,'#687582');
s.speakerNotes.textFrame.setText([
  'To investigate why AN19 favors the acoustic baseline, I used the same 5,685 available responses for MFCC, STRF and both HuBERT Tr-24 representations. I calculated mean accuracy and mean negative distance for each condition by test-recording cell. This gives 574 cells covering 48 lexical words. A recording identifies a word produced by a particular test talker. I then correlated the two cell-level quantities, giving every cell equal weight.',
  'For the second group of bars, I subtracted each lexical word\'s mean from both negative distance and accuracy before calculating Pearson correlation. STRF falls from 0.456 to 0.198, while base Tr-24 falls from 0.440 to 0.339. FT changes from 0.364 to 0.362. MFCC falls from 0.376 to 0.249. This makes word-dependent variation a more specific candidate explanation for the baseline advantage than simply saying that acoustic features are predictive.',
  'The pattern remains when I remove word and condition effects together using an ordinary least-squares projection: correlations are 0.245 for MFCC, 0.154 for STRF, 0.323 for base Tr-24 and 0.319 for FT. This projection is a descriptive sensitivity calculation and does not refit the GLMM. In X21, excluding Talker-specific still gives raw correlations of 0.094 and 0.058 for the baselines, versus 0.283 and 0.284 for HuBERT. Excluding Control as well preserves that ordering.',
  'This finding does not establish a cause or a fraction of explained variance. Centering changes the comparison, cells share words and talkers, and there is no significance test here. Word duration, phonetic composition and recording characteristics are possible explanations that remain untested. The diagnostic also uses a different predictor and statistical summary from the historical z analysis on slide 1. It motivates a matched within/between-word model comparison rather than explaining that historical z gap by itself.',
  'Sources: cross_talker_generalization/artifacts/derived/AN19-AN19_acoustic-confirmatory-full-20260906-model-input.csv; AN19-AN19_hubert_base_tsne-confirmatory-tr24-20260906-model-input.csv; AN19-AN19_hubert_ft_tsne-confirmatory-tr24-20260906-model-input.csv. Descriptive calculations and input hashes: outputs/an19_baseline_explanation_2026-09-18/.build/analyze.py and evidence.json. Additional context: cross_talker_generalization/analysis/acoustic_baselines/AN19_BASELINE_EXPLANATION.md.',
].join('\n\n'));

const candidatePath=path.join(tmp,'draft.pptx');
await(await PresentationFile.exportPptx(p)).save(candidatePath);
const finalPath=path.join(workspaceDir,'presentation/AN19_acoustic_baselines.pptx');
const result=await finalizePresentation({workspaceDir,candidatePath,finalPath,
  explicitTotalSlideCount:2,requiredNativeChartOwnerSlides:[1,2],requiredNativeTableOwnerSlides:[],
  materializeLiteralChartWorkbooks:true,
  pythonExecutable:python,integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit'],
  fontPolicy:{basis:'design',families:[font]},verifyArtifactToolImport:true,receiptPath:path.join(tmp,'validation.json')});
console.log(JSON.stringify(result));
const final=await PresentationFile.importPptx(await FileBlob.load(finalPath));
for(let i=0;i<final.slides.items.length;i++){
  const preview=await final.export({slide:final.slides.items[i],format:'png',scale:1.2});
  await fs.writeFile(path.join(tmp,`slide-${i+1}.png`),new Uint8Array(await preview.arrayBuffer()));
}
