'use strict';
const $ = s => document.querySelector(s);
const state = {config:null, active:null, models:[], encoders:[], dirty:false, pending:false, token:'', polling:false, applying:false, profile:null};
const titles = {overview:['服务概览','让动作生成，在你的掌控之中。'],models:['模型管理','管理动作模型与文本编码器，按需下载本地权重。'],inference:['推理配置','在质量、速度与显存之间，找到合适的平衡。'],server:['服务设置','设置连接地址、模型存储与任务队列。'],downloads:['下载任务','明确发起，可见进度，随时取消。']};
function element(tag, cls, text) { const el=document.createElement(tag);if(cls)el.className=cls;if(text!==undefined)el.textContent=text;return el; }
function notice(message, type='') {const el=$('#notice');el.textContent=message;el.className='notice '+type;el.hidden=!message;}
async function api(path, options={}) {
  const headers={...(options.body?{'Content-Type':'application/json'}:{}),...(state.token?{Authorization:'Bearer '+state.token}:{}),...options.headers};
  const response=await fetch(path,{...options,headers,signal:AbortSignal.timeout(15000)});
  if(!response.ok){let text;try{const body=await response.json();text=typeof body.detail==='string'?body.detail:JSON.stringify(body.detail);}catch{text=response.statusText;}
    if(response.status===401&&!$('#auth-dialog').open)$('#auth-dialog').showModal();throw Error(text||`HTTP ${response.status}`);}
  return response.json();
}
function tab(name){document.querySelectorAll('.page').forEach(el=>el.hidden=el.id!==name);document.querySelectorAll('nav button').forEach(el=>el.classList.toggle('selected',el.dataset.tab===name));$('#page-title').textContent=titles[name][0];$('#page-subtitle').textContent=titles[name][1];}
document.querySelectorAll('[data-tab]').forEach(el=>el.onclick=()=>tab(el.dataset.tab));document.querySelectorAll('[data-go]').forEach(el=>el.onclick=()=>tab(el.dataset.go));
function configState(){ $('#config-state').textContent=state.applying?'正在重启，请稍候…':state.dirty?'● 有未保存的修改':state.pending?'● 配置已保存，等待应用并重启':'✓ 配置已生效';$('#save').disabled=!state.config||state.applying||!state.dirty;$('#apply').disabled=!state.config||state.dirty||state.applying||!state.canRestart;}
function changed(){state.dirty=true;configState();}
function field(parent, key, title, opts, value, onChange){
  const label=element('label',opts.type==='checkbox'?'check':'');
  let input;
  if(opts.options){input=element('select');opts.options.forEach(o=>{const option=element('option','',typeof o==='string'?o:o[1]);option.value=typeof o==='string'?o:o[0];input.append(option);});}
  else{input=element('input');input.type=opts.type||'text';}
  input.id=key;input.name=key;
  if(opts.type==='checkbox'){input.checked=!!value;label.append(input,document.createTextNode(title));}
  else{label.append(document.createTextNode(title),input);input.value=value??'';}
  for(const a of ['min','max','step','placeholder'])if(opts[a]!==undefined)input[a]=opts[a];
  if(opts.required)input.required=true;if(opts.disabled)input.disabled=true;
  if(opts.hint)label.append(element('small','',opts.hint));
  const update=()=>{if(!input.checkValidity())return;const val=opts.type==='checkbox'?input.checked:opts.type==='number'?Number(input.value):input.value;onChange(val);};
  input.addEventListener('input',update);input.addEventListener('change',update);
  parent.append(label);return input;
}
function configField(container,key,title,opts={},group=null){const obj=group?state.config[group]:state.config;field(container,key,title,opts,obj[key],v=>{obj[key]=v;changed();});}
function forms(){
  ['inference-fields','optimization-fields','server-fields','queue-fields'].forEach(id=>$('#'+id).replaceChildren());
  const inf=$('#inference-fields'),opt=$('#optimization-fields'),server=$('#server-fields'),queue=$('#queue-fields');
  for(const [key,title,opts] of [
    ['steps','扩散步数',{type:'number',min:1,max:1000,hint:'默认 100；减少步数可能降低动作质量'}],
    ['seed','默认随机种子',{type:'number',min:0,max:2147483647}],
    ['cfg_text','文本 CFG',{type:'number',min:0,max:10,step:0.1}],
    ['cfg_constraints','约束 CFG',{type:'number',min:0,max:10,step:0.1}],
    ['transition_frames','分段过渡帧',{type:'number',min:1,max:60}],
    ['output_format','默认输出格式',{options:['bvh','npz']}],
    ['standard_tpose','标准 T 姿态',{type:'checkbox'}],
    ['post_processing','动作后处理',{type:'checkbox',hint:'需另行安装 motion_correction'}]
  ])configField(inf,key,title,opts,'inference');
  configField(opt,'block_swap','动作 block swap',{type:'number',min:0,max:128,hint:'0 关闭；SOMA v1.2 最多 32 块'});
  configField(opt,'cpu_threads','CPU 线程数',{type:'number',min:1,max:64});
  configField(opt,'offload','分阶段卸载到 CPU',{type:'checkbox',hint:'文本编码与动作采样分阶段使用显存'});
  configField(opt,'tf32','启用 TF32',{type:'checkbox',hint:'可能提高计算速度，也可能改变数值结果'});
  configField(server,'host','监听地址',{hint:'127.0.0.1 仅本机；0.0.0.0 监听所有网卡'});
  configField(server,'port','监听端口',{type:'number',min:1,max:65535,hint:'默认 6600；应用后自动跳转到新端口'});
  configField(server,'models_dir','模型根目录',{hint:'相对路径以 config.json 所在目录为基准'});
  configField(server,'data_dir','运行数据目录',{hint:'文本缓存、生成文件与任务信息'});
  configField(server,'device','计算设备',{options:['cuda','cuda:0','cpu'],hint:'ConvRot INT8 需要 CUDA'});
  configField(queue,'queue_size','最大排队数',{type:'number',min:1,max:32});
  configField(queue,'retention_hours','生成文件保留（小时）',{type:'number',min:1,max:8760});
}
function renderProfiles(){
  if(!state.config)return;
  for(const kind of ['model','encoder']){
    const entries=kind==='model'?state.config.models:state.config.encoders,available=kind==='model'?state.models:state.encoders,root=$('#'+kind+'-list');root.replaceChildren();
    for(const [id,p] of Object.entries(entries)){
      const isDefault=id===state.config['default_'+kind],status=available.find(x=>x.id===id),card=element('article','model-card'+(isDefault?' is-default':''));
      const top=element('div','topline');top.append(element('h3','',p.label||id),element('span','badge '+(status?.available?'':'dim'),status?.available?'本地就绪':'未就绪'));card.append(top);
      const tags=element('div','tags');tags.append(element('span','badge dim',kind==='model'?p.skeleton:p.kind.replace('_llm2vec','').toUpperCase()));if(p.precision)tags.append(element('span','badge dim',p.precision.toUpperCase()));if(status?.loaded)tags.append(element('span','badge','已加载'));if(status?.approximate)tags.append(element('span','badge warn','近似编码器'));card.append(tags);
      card.append(element('p','',id+'\n'+(kind==='model'?p.path:p.base)+'\n'+(p.repo_id?((p.download_source||'huggingface')==='modelscope'?'ModelScope · ':'Hugging Face · ')+p.repo_id:'本地权重，无下载仓库')));
      const actions=element('div','actions'),edit=element('button','secondary','编辑');edit.onclick=()=>editProfile(kind,id);actions.append(edit);
      const dl=element('button','secondary','↓ 下载');dl.disabled=!p.repo_id||!!status?.available;dl.onclick=()=>startDownload(kind,id);actions.append(dl);
      if(isDefault)actions.append(element('span','default-label','✓ 默认'));
      else{const def=element('button','secondary','设为默认');def.onclick=()=>{state.config['default_'+kind]=id;changed();renderProfiles();};actions.append(def);}
      card.append(actions);root.append(card);
    }
  }
}
function editProfile(kind,id=null){
  const entries=kind==='model'?state.config.models:state.config.encoders;
  const p=id?structuredClone(entries[id]):kind==='model'?{label:'新动作模型',path:'',precision:'fp32',skeleton:'SOMA',repo_id:null,download_source:'huggingface'}:{label:'新文本编码器',kind:'nf4_llm2vec',base:'',block_swap:0,download_source:'huggingface'};
  const draft={id:id||'',download_source:'huggingface',...p},root=$('#profile-fields');root.replaceChildren();state.profile={kind,id,draft};
  $('#profile-title').textContent=(id?'编辑':'添加')+(kind==='model'?'动作模型':'文本编码器');
  field(root,'profile-id','配置 ID',{required:true,disabled:!!id,hint:'用于 API 选择模型的唯一标识'},draft.id,v=>draft.id=v);
  const fields=kind==='model'?[
    ['label','显示名称',{required:true}],['path','权重目录',{required:true}],['precision','精度',{options:['fp32','fp16','bf16','int8_convrot'],hint:'FP16/BF16 复用浮点权重；INT8 需专用 ConvRot 权重'}],['skeleton','骨架类型',{options:['SOMA','SMPLX','G1']}],['download_source','下载源',{options:[['huggingface','Hugging Face'],['modelscope','ModelScope（魔搭）']]}],['repo_id','下载仓库 ID',{hint:'例如 spawner/Kimodo-SOMA-RP-v1.2-int8-convrot；本地权重可留空'}]
  ]:[['label','显示名称',{required:true}],['kind','编码器类型',{options:['nf4_llm2vec','awq_llm2vec','llm2vec']}],['base','基座权重目录',{required:true}],['download_source','下载源',{options:[['huggingface','Hugging Face'],['modelscope','ModelScope（魔搭）']],hint:'此编码器及适配器使用同一下载源'}],['repo_id','基座仓库 ID',{}],['mntp','MNTP 适配器目录',{hint:'AWQ / BF16 需要；合并 NF4 留空'}],['mntp_repo_id','MNTP 仓库 ID',{}],['supervised','Supervised 适配器目录',{}],['supervised_repo_id','Supervised 仓库 ID',{}],['block_swap','文本 block swap',{type:'number',min:0,max:32}]];
  for(const [key,title,opts] of fields)field(root,'profile-'+key,title,opts,draft[key],v=>draft[key]=v);
  $('#profile-delete').hidden=!id;$('#profile-dialog').showModal();
}
$('#profile-close').onclick=()=>$('#profile-dialog').close();
$('#profile-form').onsubmit=event=>{event.preventDefault();const {kind,id,draft}=state.profile;const entries=kind==='model'?state.config.models:state.config.encoders;const key=draft.id.trim();
  if(!key||(!id&&entries[key])){notice('配置 ID 不能为空或重复','error');return;}
  const p=structuredClone(draft);delete p.id;for(const k of ['repo_id','mntp','supervised','mntp_repo_id','supervised_repo_id'])if(p[k]==='')p[k]=null;
  if(kind==='encoder'&&p.kind!=='nf4_llm2vec'&&(!p.mntp||!p.supervised)){notice('AWQ / BF16 编码器需要两组适配器路径','error');return;}
  entries[key]=p;changed();renderProfiles();$('#profile-dialog').close();};
$('#profile-delete').onclick=()=>{const {kind,id}=state.profile;if(state.config['default_'+kind]===id){notice('请先选择其他默认配置，再删除此配置','error');return;}delete (kind==='model'?state.config.models:state.config.encoders)[id];changed();renderProfiles();$('#profile-dialog').close();};
$('#add-model').onclick=()=>editProfile('model');$('#add-encoder').onclick=()=>editProfile('encoder');
function summary(){
  const a=state.active;const dl=$('#active-summary');dl.replaceChildren();
  for(const [k,v] of [['默认动作模型',a.default_model],['默认文本编码器',a.default_encoder],['计算设备',a.device.toUpperCase()],['分阶段卸载',a.offload?'开启':'关闭'],['动作 block swap',a.block_swap],['默认扩散步数',a.inference?.steps??100]])dl.append(element('dt','',k),element('dd','',String(v)));
}
async function loadConfig(){const data=await api('/v1/admin/config');state.config=data.saved;state.active=data.active;
  state.config.inference={...state.active.inference,...state.config.inference};state.canRestart=data.can_restart;state.pending=data.pending_restart;state.dirty=false;
  $('#config-path').textContent=data.config_path||'未绑定配置文件';forms();summary();configState();renderProfiles();}
async function save(){try{$('#save').disabled=true;const result=await api('/v1/admin/config',{method:'PUT',body:JSON.stringify(state.config)});state.dirty=false;state.pending=result.pending_restart;configState();notice('配置已保存。点击“应用并重启”使修改生效。','success');}catch(error){notice(error.message,'error');configState();}}
$('#save').onclick=save;
$('#apply').onclick=async()=>{try{state.applying=true;configState();const result=await api('/v1/admin/apply',{method:'POST'});let target=new URL(location.href);if(!['0.0.0.0','::','127.0.0.1','localhost','::1'].includes(result.host))target.hostname=result.host;target.port=result.port;target.pathname='/';
    notice('正在应用配置并重启。恢复后将自动打开 '+target.origin+'；Blender 端请同步连接此地址。');
    if(target.origin!==location.origin){setTimeout(()=>location.assign(target.href),5000);return;}
    for(let attempt=0;attempt<60;attempt++){await new Promise(resolve=>setTimeout(resolve,1500));try{await loadConfig();state.applying=false;configState();await poll();notice('配置已生效，服务已恢复。','success');return;}catch{}}
    state.applying=false;configState();notice('等待服务恢复超时，请查看服务终端日志并刷新页面。','error');
  }catch(error){state.applying=false;configState();notice(error.message,'error');}};
async function startDownload(kind,id){
  if(state.dirty||state.pending){notice('请先保存并应用配置，再下载模型。','error');return;}
  const profile=(kind==='model'?state.config.models:state.config.encoders)[id];
  if(!confirm('下载 '+profile.label+'？\n仓库：'+profile.repo_id+'\n权重可能占用数 GB；下载可取消并续传。'))return;
  try{await api('/v1/admin/downloads',{method:'POST',body:JSON.stringify({kind,profile:id,endpoint:$('#download-endpoint').value,token:$('#download-token').value||null,modelscope_token:$('#modelscope-token').value||null})});$('#download-token').value='';$('#modelscope-token').value='';tab('downloads');notice('下载已开始。','success');await poll();}catch(error){notice(error.message,'error');}
}
function jobs(root,items,download=false){
  root.replaceChildren();root.classList.toggle('empty',!items.length);if(!items.length){root.textContent=download?'暂无下载任务':'暂无任务。在 Blender 中提交动作生成后，进度会显示在这里。';return;}
  const labels={queued:'排队中',running:'运行中',cancelling:'取消中',succeeded:'已完成',failed:'失败',cancelled:'已取消'};
  for(const j of items){const row=element('div','job-row'),top=element('div','job-top'),left=element('div');left.append(element('span','',download?j.profile:(j.kind==='load'?'加载模型':'生成动作')+' · '+j.id.slice(0,8)),element('span','badge '+(j.state==='failed'?'warn':'dim'),labels[j.state]||j.state));top.append(left);
    const actions=element('div','job-actions');if(['queued','running','cancelling'].includes(j.state)){const cancel=element('button','secondary','取消');cancel.disabled=j.state==='cancelling';cancel.onclick=async()=>{try{await api(download?'/v1/admin/downloads/'+j.id+'/cancel':'/v1/jobs/'+j.id+'/cancel',{method:'POST'});await poll();}catch(e){notice(e.message,'error');}};actions.append(cancel);}
    actions.append(element('time','',new Date(j.created_at*1000).toLocaleTimeString('zh-CN',{hour12:false})));top.append(actions);row.append(top,element('p','',j.error||j.message));
    if(['running','queued','cancelling'].includes(j.state)){const progress=element('progress');progress.max=1;if(j.progress!==null)progress.value=j.progress;progress.setAttribute('aria-label','任务进度');row.append(progress);}
    root.append(row);
  }
}
async function poll(){if(state.polling||state.applying)return;state.polling=true;
  try{const [health,models,encoders,status]=await Promise.all([api('/v1/health'),api('/v1/models'),api('/v1/encoders'),api('/v1/admin/status')]);
    state.models=models.models;state.encoders=encoders.encoders;$('#connection-dot').classList.add('online');$('#connection-text').textContent='服务已连接';$('#version').textContent='API v1 · '+health.service_version;$('#stat-status').textContent='运行中';$('#stat-address').textContent=location.host;$('#stat-models').textContent=state.models.filter(x=>x.available).length+' / '+state.models.length;$('#stat-encoders').textContent=state.encoders.filter(x=>x.available).length+' / '+state.encoders.length;$('#model-count').textContent=state.models.length;$('#stat-jobs').textContent=status.jobs.filter(j=>['queued','running','cancelling'].includes(j.state)).length;$('#download-count').textContent=status.downloads.filter(j=>['running','cancelling'].includes(j.state)).length;jobs($('#inference-jobs'),status.jobs);jobs($('#download-list'),status.downloads,true);renderProfiles();
  }catch(error){$('#connection-dot').classList.remove('online');$('#connection-text').textContent='连接已断开';$('#stat-status').textContent='离线';if(!state.config)notice('无法读取服务配置：'+error.message,'error');}finally{state.polling=false;}
}
$('#refresh').onclick=async()=>{try{if(!state.config||!state.dirty){await loadConfig();}await poll();}catch(e){notice(e.message,'error');}};
$('#auth-open').onclick=()=>$('#auth-dialog').showModal();$('#auth-close').onclick=()=>$('#auth-dialog').close();
$('#auth-form').onsubmit=async e=>{e.preventDefault();state.token=$('#api-token').value;$('#api-token').value='';$('#auth-dialog').close();try{await loadConfig();await poll();notice('连接成功','success');}catch(error){notice(error.message,'error');}};
window.addEventListener('beforeunload',e=>{if(state.dirty){e.preventDefault();e.returnValue='';}});
(async()=>{try{await loadConfig();}catch(error){notice('无法读取配置：'+error.message,'error');}await poll();setInterval(poll,2000);})();
