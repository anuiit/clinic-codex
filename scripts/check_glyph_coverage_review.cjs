const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),assert=require('node:assert/strict');
const {chromium}=require('C:/Users/anmou/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const dest='C:/Users/anmou/.codex/visualizations/2026/10/02/01a0fee0-5bd2-70f2-8f5d-f80a2437891c/glyph-coverage-review';
(async()=>{
 const html=fs.readFileSync(path.join(dest,'index.html'),'utf8');for(const m of html.matchAll(/<script>([\s\S]*?)<\/script>/g))new vm.Script(m[1]);
 const browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true,args:['--allow-file-access-from-files']});
 try{
  const tab=await browser.newPage({viewport:{width:1450,height:1120}}),errors=[];tab.on('pageerror',e=>errors.push(e.message));
  await tab.goto('file:///'+dest+'/index.html');await tab.locator('#selectedTitle').filter({hasText:'Tête S14 confirmée'}).waitFor();
  assert.match(await tab.locator('#progress').innerText(),/44 suggestions/);
  await tab.screenshot({path:path.join(dest,'review-preview.png'),fullPage:true});
  await tab.locator('#accept').click();assert.match(await tab.locator('#status').innerText(),/relecteur/);
  await tab.locator('#reviewer').fill('QA — simulation sans validation métier');
  await tab.locator('#class').fill('classe_inexistante');await tab.locator('#accept').click();assert.match(await tab.locator('#status').innerText(),/catalogue/);
  await tab.locator('#class').fill('tlacatl');await tab.locator('#x').fill('99999');await tab.locator('#accept').click();assert.match(await tab.locator('#status').innerText(),/intérieur/);
  await tab.locator('#x').fill('90');await tab.locator('#accept').click();assert.match(await tab.locator('#progress').innerText(),/43 suggestions/);
  await tab.locator('#coverage').selectOption('complete');assert.match(await tab.locator('#status').innerText(),/Inspectez/);
  await tab.locator('#inspected').check();await tab.locator('#coverage').selectOption('complete');assert.match(await tab.locator('#status').innerText(),/43 suggestions/);
  assert.equal(await tab.locator('#coverage').inputValue(),'unreviewed');assert.match(await tab.locator('#coverageSummary').innerText(),/Aucune attestation/);
  await tab.locator('#export').click();const first=JSON.parse(await tab.locator('#exportText').inputValue());const packet=JSON.parse(fs.readFileSync(path.join(dest,'packet.json'),'utf8')),p=packet.pages[0],r=first.pages[p.group];
  assert.equal(r.decisions['confirmed-s14-head'].status,'accepted');assert.equal(r.coverage_status,'unreviewed');assert.equal(r.reviewed_at,null);
  await tab.locator('#closeExport').click();
  // A full-page claim with a newly duplicated known box must fail and restore all metadata.
  for(const c of p.candidates)if(c.id!=='confirmed-s14-head')r.decisions[c.id]={status:'rejected',reviewed_by:first.reviewer,reviewed_at:new Date().toISOString()};
  r.duplicate_resolutions[p.duplicate_pairs[0].id]='distinct';
  const original=p.known.find(z=>!p.duplicate_pairs.some(d=>d.a===z.id||d.b===z.id));
  r.manual_items.push({id:'manual-qa',kind:'candidate',bbox:original.bbox,suggested_class:original.class_name});
  r.decisions['manual-qa']={status:'accepted',bbox:original.bbox,class_name:original.class_name,reviewed_by:first.reviewer,reviewed_at:new Date().toISOString()};
  const fixture=path.join(dest,'qa-fixture.json');fs.writeFileSync(fixture,JSON.stringify(first));await tab.locator('#import').setInputFiles(fixture);
  await tab.locator('#status').filter({hasText:'Décisions reprises'}).waitFor();await tab.locator('#inspected').check();await tab.locator('#coverage').selectOption('complete');
  assert.match(await tab.locator('#status').innerText(),/nouvelle paire/);assert.equal(await tab.locator('#coverage').inputValue(),'unreviewed');assert.match(await tab.locator('#coverageSummary').innerText(),/Aucune attestation/);
  const duplicate=tab.locator('#duplicates .duplicate').filter({hasText:'manual-qa'});await duplicate.locator('select').selectOption('distinct');
  await tab.locator('#inspected').check();await tab.locator('#coverage').selectOption('complete');assert.equal(await tab.locator('#coverage').inputValue(),'complete');
  await tab.locator('#export').click();const complete=JSON.parse(await tab.locator('#exportText').inputValue());assert.equal(complete.pages[p.group].coverage_status,'complete');
  fs.writeFileSync(path.join(dest,'qa-export-example.json'),JSON.stringify(complete,null,2));await tab.locator('#closeExport').click();
  await tab.locator('#page').selectOption('1');await tab.locator('#progress').filter({hasText:'2/17'}).waitFor();
  assert.deepEqual(errors,[]);
  fs.writeFileSync(path.join(dest,'qa-results.json'),JSON.stringify({passed:true,javascript_parse:true,initial44_candidates:true,missing_reviewer_rejected:true,invalid_class_bbox_rejected:true,accepted_head_exported:true,pending_blocks_complete:true,dynamic_duplicate_blocks_complete:true,failed_attestation_metadata_restored:true,explicit_distinct_resolution_passes:true,page_navigation:true,console_errors:errors,simulation_only:true},null,2));
  console.log('UI_CHECK_PASSED: initial display, validation, import/export, completeness guard, dynamic duplicate and navigation');
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
