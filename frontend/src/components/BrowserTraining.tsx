import {useEffect,useMemo,useRef,useState} from 'react';
import {IconActivity,IconAlertTriangle,IconCheck,IconCpu,IconDownload,IconPlayerPlay,IconPlayerStop} from '@tabler/icons-react';
import {Plot,trace} from '@/components/Plot';
import {number,percent} from '@/lib/api';
import {roadsterFlightFigure} from '@/lib/roadster-figure';
import type {BrowserEpisode,RewardVariant} from '@/lib/browser-ppo';

type WorkerMessage={type:'episode';episode:BrowserEpisode;updates:number}|{type:'complete'|'stopped';policy:unknown}|{type:'status';message:string}|{type:'error';message:string};
const rewards:Record<RewardVariant,{label:string;description:string}>= {
  baseline:{label:'Task reward',description:'Optimizes the original landing objective.'},
  upright_bonus:{label:'Upright dwell probe',description:'Adds +1 while upright and nearly motionless, even if the car simply stays on the ramp.'},
  potential_shaping:{label:'Potential progress',description:'Adds discounted potential-difference progress toward the pad and a controlled touchdown.'},
};

export function BrowserTraining(){
  const workerRef=useRef<Worker|null>(null);
  const [variant,setVariant]=useState<RewardVariant>('baseline'),[seed,setSeed]=useState(7),[target,setTarget]=useState(20);
  const [episodes,setEpisodes]=useState<BrowserEpisode[]>([]),[selected,setSelected]=useState(0),[running,setRunning]=useState(false);
  const [updates,setUpdates]=useState(0),[status,setStatus]=useState('Ready when you are.'),[error,setError]=useState(''),[policy,setPolicy]=useState<unknown>(null);
  useEffect(()=>()=>workerRef.current?.terminate(),[]);
  const current=episodes.find(item=>item.episode===selected)||episodes[episodes.length-1];
  const replay=useMemo(()=>current?roadsterFlightFigure(current.flight):null,[current]);
  const failures=useMemo(()=>episodes.filter(item=>!item.safe),[episodes]);
  const avgReturn=episodes.length?episodes.reduce((sum,item)=>sum+item.nativeReturn,0)/episodes.length:null;
  const safeRate=episodes.length?episodes.filter(item=>item.safe).length/episodes.length:null;
  const rewardCurve=useMemo(()=>[
    trace(episodes.map(item=>item.episode),episodes.map(item=>item.nativeReturn),'Environment return','#eeeeee'),
    trace(episodes.map(item=>item.episode),episodes.map(item=>item.shapedReturn),'Training return','#858585','dash'),
  ],[episodes]);
  const outcomeCurve=useMemo(()=>[
    trace(episodes.map(item=>item.episode),episodes.map((_,i)=>episodes.slice(Math.max(0,i-9),i+1).filter(e=>e.safe).length/episodes.slice(Math.max(0,i-9),i+1).length*100),'Rolling safe landing rate (%)','#eeeeee'),
  ],[episodes]);
  const start=()=>{
    if(workerRef.current)return;
    setError('');setEpisodes([]);setSelected(0);setUpdates(0);setPolicy(null);setRunning(true);setStatus('Creating the in-browser policy and simulator…');
    const worker=new Worker(new URL('../lib/browser-ppo.worker.ts',import.meta.url),{type:'module'});workerRef.current=worker;
    worker.onmessage=(event:MessageEvent<WorkerMessage>)=>{
      const message=event.data;
      if(message.type==='episode'){
        setEpisodes(previous=>[...previous,message.episode]);setSelected(message.episode.episode);setUpdates(message.updates);
        setStatus(`PPO update ${message.updates} complete · ${message.episode.outcome.toLowerCase()}`);
      }else if(message.type==='status'){
        setStatus(message.message);
      }else if(message.type==='complete'||message.type==='stopped'){
        setPolicy(message.policy);setRunning(false);setStatus(message.type==='stopped'?'Training stopped. The partial policy is ready to download.':'Training complete. The browser policy is ready to download.');
        worker.terminate();workerRef.current=null;
      }else if(message.type==='error'){
        setError(message.message);setRunning(false);setStatus('The browser run stopped with an error.');worker.terminate();workerRef.current=null;
      }
    };
    worker.onerror=event=>{setError(event.message||'The browser worker could not start.');setRunning(false);setStatus('The browser run stopped with an error.');worker.terminate();workerRef.current=null;};
    worker.postMessage({type:'start',variant,seed,episodes:target});
  };
  const stop=()=>{workerRef.current?.postMessage({type:'stop'});setStatus('Stopping after the current PPO update…');};
  const download=()=>{
    if(!policy)return;
    const url=URL.createObjectURL(new Blob([JSON.stringify(policy,null,2)],{type:'application/json'}));
    const link=document.createElement('a');link.href=url;link.download=`browser-ppo-${variant}-seed-${seed}.json`;link.click();window.setTimeout(()=>URL.revokeObjectURL(url),1000);
  };
  const clear=()=>{if(running)return;setEpisodes([]);setSelected(0);setUpdates(0);setPolicy(null);setError('');setStatus('Ready when you are.');};

  return <section className="browser-training" aria-labelledby="browser-training-title">
    <div className="browser-training-intro">
      <div><span className="eyebrow"><span className="tiny-square"/>ON-DEVICE TRAINING · NO SERVER JOB</span>
        <h2 id="browser-training-title">Run a small PPO experiment in this tab.</h2>
        <p>This optional sandbox runs a compact PPO actor and value network on your device. Each attempt uses Rapier3D rigid-body physics on a collidable Tesla-style ramp with a broad curved foot, steep straight climb, and short curved shoulder, plus gravity, ground contact, air drag, wind, a learned powered run-up, and six flight controls.</p>
      </div>
      <div className="browser-local-badge"><IconCpu size={16}/><span>COMPUTE STAYS IN THIS BROWSER</span></div>
    </div>
    <div className="roadster-note browser-demo-note"><IconAlertTriangle size={16}/><span><strong>v6 curved-top ramp experiment:</strong> the foot transition, near-vertical stem, and upper shoulder form a triangle-mesh collider in both the Rapier3D WebAssembly and Python environments. Gravity and contact forces produce the launch; the path is not scripted. This remains a simplified rigid-body proxy, not a validated Tesla model. Short runs demonstrate learning updates; they do not establish a solved task. Reloading clears this browser session.</span></div>
    {error&&<div className="notice" role="alert">{error}<button onClick={()=>setError('')} aria-label="Dismiss error">×</button></div>}

    <div className="browser-train-grid">
      <section className="surface browser-train-controls">
        <span className="eyebrow">EXPERIMENT SETUP</span><h3>Choose a reward and budget.</h3>
        <label className="browser-field"><span>REWARD DESIGN</span><select value={variant} onChange={event=>setVariant(event.target.value as RewardVariant)} disabled={running}>
          {(Object.keys(rewards) as RewardVariant[]).map(key=><option value={key} key={key}>{rewards[key].label}</option>)}
        </select><small>{rewards[variant].description}</small></label>
        <div className="browser-field-row">
          <label className="browser-field"><span>RANDOM SEED</span><input type="number" min="0" max="2147483647" value={seed} onChange={event=>setSeed(Math.max(0,Number(event.target.value)||0))} disabled={running}/></label>
          <label className="browser-field"><span>TRAINING EPISODES</span><select value={target} onChange={event=>setTarget(Number(event.target.value))} disabled={running}><option value={10}>10 · quick</option><option value={20}>20 · standard</option><option value={50}>50 · extended</option></select></label>
        </div>
        <div className="browser-action-row">
          {!running?<button className="button browser-run-button" onClick={start}><IconPlayerPlay size={15}/>Start browser training</button>:<button className="button browser-run-button" onClick={stop}><IconPlayerStop size={15}/>Stop after update</button>}
          <button className="button ghost" onClick={clear} disabled={running||!episodes.length}>Reset</button>
          {policy!==null&&<button className="button ghost browser-download" onClick={download}><IconDownload size={15}/>Download policy</button>}
        </div>
        <div className="browser-progress"><div><span>{status}</span><b>{episodes.length}/{target}</b></div><div className="browser-progress-track"><i style={{width:`${Math.min(100,episodes.length/target*100)}%`}}/></div></div>
        <div className="browser-method"><strong>What is actually trained?</strong><p>A 16 → 24 → 7 tanh Gaussian actor and a 16 → 24 → 1 value network. PPO controls ramp drive from rest and then forward, vertical, and lateral thrust plus pitch, roll, and yaw jets. It records launches and flights, computes GAE advantages, then applies clipped updates with Adam. Weights stay in this tab unless downloaded.</p></div>
        <div className="browser-policy-stats"><span><small>POLICY UPDATES</small><strong>{updates}</strong></span><span><small>ENGINE</small><strong>Rapier3D · WASM</strong></span></div>
      </section>

      <section className="surface browser-training-stage">
        <div className="panel-heading"><div><span className="eyebrow">LATEST BROWSER ROLLOUT</span><h3>{current?`Episode ${current.episode} · ${current.outcome}`:'Waiting for the first episode'}</h3></div><span className="pill"><IconActivity size={13}/>LOCAL PPO</span></div>
        {replay?<Plot key={`browser-${current?.episode}`} label="Three-dimensional replay of the selected browser training episode" data={replay.data} customLayout={replay.layout} frames={replay.frames} className="roadster-plot browser-training-plot"/>:
          <div className="flight-empty browser-empty"><div className="viewer-orbit"><IconCheck size={23}/></div><h3>Every attempt is recorded.</h3><p>Start training to see this browser’s own trajectory, including failed landings.</p></div>}
        <div className="viewer-hints"><span>PPO controls launch and landing</span><span>Rapier gravity & contacts</span><span>Failure replay from this episode</span></div>
      </section>
    </div>

    <div className="browser-metrics">
      {[{name:'Episodes',value:number(episodes.length,0)},{name:'Safe landing rate',value:safeRate==null?'—':percent(safeRate)},{name:'Mean environment return',value:number(avgReturn)},{name:'Failures',value:number(failures.length,0)}].map(metric=><div className="roadster-metric surface" key={metric.name}><strong>{metric.value}</strong><span>{metric.name}</span></div>)}
    </div>
    <div className="two-columns browser-charts">
      <section className="surface"><span className="eyebrow">EPISODE RETURNS</span><h3>Task reward and training reward</h3><p className="muted small">A gap appears when the selected shaping reward differs from the environment objective.</p>{episodes.length?<Plot label="Browser training return compared with original environment return episode by episode" data={rewardCurve} customLayout={{showlegend:true,xaxis:{title:{text:'Training episode'},gridcolor:'#252525'},yaxis:{title:{text:'Episode return'},gridcolor:'#252525'}}}/>:<div className="chart-empty">Returns appear after the first rollout.</div>}</section>
      <section className="surface"><span className="eyebrow">LANDING OUTCOMES</span><h3>Rolling safe landing rate</h3><p className="muted small">Ten-episode window. Short browser runs have high statistical variability.</p>{episodes.length?<Plot label="Rolling ten episode safe landing rate during browser training" data={outcomeCurve} customLayout={{xaxis:{title:{text:'Training episode'},gridcolor:'#252525'},yaxis:{title:{text:'Safe landings (%)'},gridcolor:'#252525',range:[0,100]}}}/>:<div className="chart-empty">Outcome rate appears after the first rollout.</div>}</section>
    </div>

    <div className="section-heading browser-failure-heading"><div><span className="eyebrow">THIS BROWSER RUN</span><h2>Training attempts and failures.</h2></div><span className="muted small">{failures.length} failed · {episodes.length} total episodes</span></div>
    <section className="surface browser-attempts">
      {!episodes.length?<div className="chart-empty">No browser episodes yet. Saved Python failures remain available in the saved experiment view.</div>:<div className="table-scroll"><table><thead><tr><th>Episode</th><th>Outcome</th><th>Task return</th><th>Training return</th><th>Steps</th><th></th></tr></thead><tbody>{episodes.map(item=><tr key={item.episode} className={selected===item.episode?'active-row':''}><td className="mono">{item.episode}</td><td><span className={item.safe?'browser-safe-outcome':'failure-outcome'}>{item.outcome}</span></td><td className="mono">{number(item.nativeReturn)}</td><td className="mono">{number(item.shapedReturn)}</td><td className="mono">{item.length}</td><td><button className="review-attempt" onClick={()=>setSelected(item.episode)}>Show 3D replay ↗</button></td></tr>)}</tbody></table></div>}
    </section>
    <p className="fine-print browser-fine-print">This is a small browser-local learning exercise using Rapier3D. Its custom PPO optimizer and network differ from the Python/SB3 experiment. The older saved benchmark uses simulator v3; these v6 metrics must stay separate from it.</p>
  </section>;
}
