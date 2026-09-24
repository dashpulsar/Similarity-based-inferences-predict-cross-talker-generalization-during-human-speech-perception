// Read-only browser QA of this task's local HTML; no network or account access.
const {chromium}=require('C:/Users/Alex/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const {pathToFileURL}=require('url');
const path=require('path');
const fs=require('fs');
(async()=>{
  const browser=await chromium.launch({headless:true,channel:'msedge'});
  const page=await browser.newPage({viewport:{width:1440,height:1050}});
  const errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.route(/^https?:/,r=>r.abort());
  await page.goto(pathToFileURL(path.join(__dirname,'x21_raw_exponential.html')).href);
  await page.waitForFunction(()=>document.querySelector('#raw-landscape')?._fullLayout?.scene);
  await page.screenshot({path:path.join(__dirname,'html_preview.png'),fullPage:true});
  const before=await page.evaluate(()=>{
    const p=document.querySelector('#raw-landscape');return {traces:p.data.length,title:p.layout.scene.zaxis.title.text,z:JSON.stringify(p.data[1].z)};
  });
  await page.getByText('Signed-log z display',{exact:true}).click();
  await page.waitForFunction(()=>document.querySelector('#raw-landscape').layout.scene.zaxis.title.text.includes('log10'));
  await page.getByText('Linear z axis',{exact:true}).click();
  await page.waitForFunction(()=>document.querySelector('#raw-landscape').layout.scene.zaxis.title.text==='Signed training Wald z');
  const after=await page.evaluate(()=>{
    const p=document.querySelector('#raw-landscape');return {traces:p.data.length,title:p.layout.scene.zaxis.title.text,z:JSON.stringify(p.data[1].z)};
  });
  await page.evaluate(()=>Plotly.relayout(document.querySelector('#raw-landscape'),{'scene.camera':{eye:{x:1.8,y:1.2,z:.9}}}));
  await page.waitForFunction(()=>Math.abs(document.querySelector('#raw-landscape').layout.scene2.camera.eye.x-1.8)<1e-6);
  const qa={errors,before_title:before.title,after_title:after.title,trace_count:after.traces,score_array_restored:before.z===after.z,camera_sync:true};
  if(errors.length || !qa.score_array_restored || after.traces!==6)throw Error(JSON.stringify(qa));
  fs.writeFileSync(path.join(__dirname,'html_qa.json'),JSON.stringify(qa,null,2));
  console.log(JSON.stringify(qa));await browser.close();
})().catch(e=>{console.error(e);process.exitCode=1;});
