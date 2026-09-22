'use strict';
const $ = s => document.querySelector(s);
const fmt = (n,d=5) => (n>0?'+':'')+n.toFixed(d);
const dialog=$('#figure-dialog');
document.querySelectorAll('[data-zoom]').forEach(button=>button.addEventListener('click',()=>{
  $('#enlarged-figure').src=button.dataset.zoom;
  $('#enlarged-figure').alt=button.dataset.label;
  $('#figure-dialog-title').textContent=button.dataset.label;
  dialog.showModal();
}));
$('#close-figure').addEventListener('click',()=>dialog.close());
dialog.addEventListener('click',e=>{if(e.target===dialog)dialog.close();});
document.querySelectorAll('[data-copy]').forEach(button=>button.addEventListener('click',async()=>{
  try{await navigator.clipboard.writeText(document.getElementById(button.dataset.copy).textContent);const label=button.textContent;button.textContent='Copied ✓';$('#copy-status').textContent='Copied to clipboard.';setTimeout(()=>button.textContent=label,2000);}
  catch{const selection=window.getSelection();const range=document.createRange();range.selectNodeContents(document.getElementById(button.dataset.copy));selection.removeAllRanges();selection.addRange(range);$('#copy-status').textContent='Text selected. Use Ctrl+C or ⌘C to copy.';}
}));
async function initEvidence(){
  const response=await fetch('assets/results.json');
  if(!response.ok)throw new Error('Evidence unavailable');
  const data=await response.json();
  const candidates=data.candidate_audit.paired_points.map(p=>{
    const points=data.candidate_audit.points.filter(q=>q.case_id===p.case_id);
    const invariant=points.every(q=>q.prediction_rms===0);
    return {...p,points,status:p.positive_scaffold_ci95_both_folds?'positive':invariant?'invariant':'sensitive'};
  });
  let selected=candidates[0].case_id;
  function renderDetail(c){
    selected=c.case_id;
    document.querySelectorAll('.candidate').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.id===selected)));
    $('#candidate-detail').innerHTML=`<span class="eyebrow">${c.dataset.toUpperCase()} · ${c.case_id.replace('source_','SOURCE ')}</span><h3>${c.status==='positive'?'Positive on both folds':c.status==='invariant'?'Compound-invariant':'Sensitive to replacement'}</h3><p>Source path: <strong>${c.source_status}</strong></p><table><caption>Target-loss increase [95% CI]</caption><tbody>${c.points.map(p=>`<tr><th>Fold ${p.fold}</th><td>${fmt(p.target_loss_gain)}<small>[${fmt(p.target_loss_gain_ci95_lower)}, ${fmt(p.target_loss_gain_ci95_upper)}]</small></td></tr>`).join('')}</tbody></table><p class="caption">Prediction RMS change: ${c.points.map(p=>`F${p.fold} ${p.prediction_rms.toFixed(5)}`).join(' · ')}.<br>A positive loss increase favors the original compound input over its registered replacements.</p>`;
  }
  function renderCandidates(cohort){
    const subset=candidates.filter(c=>cohort==='all'||c.dataset===cohort);
    $('#candidate-grid').innerHTML=subset.map(c=>`<button class="candidate ${c.status}" data-id="${c.case_id}" aria-pressed="false" aria-label="${c.dataset.toUpperCase()}, source ${Number(c.case_id.split('_')[1])}, ${c.status}">${Number(c.case_id.split('_')[1])}</button>`).join('');
    $('#candidate-grid').querySelectorAll('button').forEach(b=>b.addEventListener('click',()=>renderDetail(candidates.find(c=>c.case_id===b.dataset.id))));
    $('#cohort-count').textContent=`${subset.length} sources · ${subset.filter(c=>c.status==='positive').length} positive on both folds · ${subset.filter(c=>c.status==='sensitive').length} other sensitive · ${subset.filter(c=>c.status==='invariant').length} invariant`;
    renderDetail(subset.find(c=>c.case_id===selected)||subset[0]);
  }
  document.querySelectorAll('[data-cohort]').forEach(b=>b.addEventListener('click',()=>{document.querySelectorAll('[data-cohort]').forEach(c=>c.setAttribute('aria-pressed',String(c===b)));renderCandidates(b.dataset.cohort);}));
  function renderFeedback(){
    const metric=$('#feedback-metric').value.replace('chemical_loss_gain','chemical_target_loss_gain').replace('dose_loss_gain','dose_target_loss_gain');
    const p=data.feedback.paired_effects.find(p=>p.boundary===$('#feedback-fold').value&&p.metric===metric);
    const span=Math.max(Math.abs(p.ci95_low),Math.abs(p.ci95_high))*1.25;
    const x=n=>60+(n+span)/(2*span)*760;
    $('#feedback-chart').innerHTML=`<div class="effect-heading"><strong>${fmt(p.mean,6)}</strong><span>Mean paired difference<br><b>95% CI [${fmt(p.ci95_low,6)}, ${fmt(p.ci95_high,6)}]</b></span></div><svg viewBox="0 0 880 145" role="img" aria-label="Paired mean difference ${fmt(p.mean,6)} with 95 percent interval from ${fmt(p.ci95_low,6)} to ${fmt(p.ci95_high,6)}, spanning zero"><line x1="60" y1="70" x2="820" y2="70" stroke="#c5d4d6"/><line x1="440" y1="20" x2="440" y2="102" stroke="#83939b" stroke-dasharray="4 4"/><line x1="${x(p.ci95_low)}" y1="70" x2="${x(p.ci95_high)}" y2="70" stroke="#176b73" stroke-width="6"/><line x1="${x(p.ci95_low)}" y1="57" x2="${x(p.ci95_low)}" y2="83" stroke="#176b73" stroke-width="2"/><line x1="${x(p.ci95_high)}" y1="57" x2="${x(p.ci95_high)}" y2="83" stroke="#176b73" stroke-width="2"/><circle cx="${x(p.mean)}" cy="70" r="9" fill="#176b73" stroke="white" stroke-width="3"/><g font-family="Segoe UI, sans-serif" font-size="16" fill="#5b6b78"><text x="60" y="127">${fmt(-span,4)}</text><text x="440" y="127" text-anchor="middle">0</text><text x="820" y="127" text-anchor="end">${fmt(span,4)}</text></g></svg><div class="effect-note">${metric==='mse'?'Lower MSE favors audit feedback.':'Higher values favor audit feedback.'} ${p.wins} of ${p.n} paired trajectories favor audit feedback for this metric.</div>`;
  }
  $('#feedback-fold').addEventListener('change',renderFeedback);
  $('#feedback-metric').addEventListener('change',renderFeedback);
  renderCandidates('all');renderFeedback();
}
initEvidence().catch(()=>{
  $('#candidate-detail').textContent='See the complete candidate audit in the figure below.';
  $('#feedback-chart').innerHTML='<p>See paired feedback results and uncertainty in the figure below, or <a href="assets/results.json">download the data</a>.</p>';
});
