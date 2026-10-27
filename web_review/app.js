const $ = id => document.getElementById(id);
let reports = [];
const escape = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function render() {
  const visible = reports.filter(r => $('severity').value === 'All' || r.severity === $('severity').value);
  $('count').textContent = visible.length + ' incident(s) shown';
  $('reports').innerHTML = visible.map(r => `<article data-report="${r.id}"><h3>${escape(r.category)} · ${escape(r.severity)}</h3><p>${escape(r.summary)}</p><p>${escape(r.track_id)} · ${escape(r.source)} · ${escape(r.date)} ${escape(r.time)}</p><p class="state">${r.dismissed ? 'Dismissed' : r.acknowledged ? 'Acknowledged' : 'Needs review'}</p><div class="actions"><button data-action="ack" data-id="${r.id}">Acknowledge</button><button data-action="dismiss" data-id="${r.id}">Dismiss</button></div></article>`).join('');
  $('download').disabled = !reports.length;
}
async function load() {
  $('error').textContent = '';
  reports = []; render();
  try {
    const response = await fetch('/api/preview', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(JSON.parse($('source').value))});
    const result = await response.json();
    if (!response.ok) throw Error(result.error);
    reports = result.reports; render();
  } catch (error) { $('error').textContent = 'Could not load report: ' + error.message; }
}
$('load').onclick = load;
$('sample').onclick = async () => {try {$('source').value = JSON.stringify(await (await fetch('/sample.json')).json(),null,2); await load();} catch(e) {$('error').textContent=e.message;}};
$('file').onchange = async () => {const file=$('file').files[0];if(!file)return;if(file.size>1048576){$('error').textContent='Report exceeds 1 MB.';return;}$('source').value=await file.text();await load();};
$('severity').onchange = render;
$('reports').onclick = e => {const button=e.target.closest('[data-action]');if(!button)return;const row=reports.find(r=>r.id===Number(button.dataset.id));if(button.dataset.action==='ack'){row.acknowledged=true;row.dismissed=false;}else row.dismissed=true;render();};
$('download').onclick = () => {const url=URL.createObjectURL(new Blob([JSON.stringify(reports,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='reviewed-incidents.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
