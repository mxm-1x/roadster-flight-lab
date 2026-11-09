import {useEffect,useMemo,useState} from 'react';
import {Plot,trace} from '@/components/Plot';
import {number,percent} from '@/lib/api';
import {roadsterFlightFigure,type SavedRoadster} from '@/lib/roadster-figure';

type Attempt={episode:number;outcome:string;native_return:number;shaped_return:number;flight_url?:string;bundle_url?:string;bundle_index?:number};
type Run={variant:string;label:string;seed:number;simulator_version:number;stage:string;policy_type:string;timesteps:number;summary:{episodes:number;safe_landing_rate:number;crash_rate:number;timeout_rate:number;mean_native_return:number};evaluations:Attempt[];training_flights:Attempt[];training_curve:{timesteps:number;return:number|null;length:number|null}[]};
type Index={simulator_versions:number[];runs:Run[]};
const keyOf=(run:Run)=>`${run.simulator_version}:${run.stage}:${run.variant}:${run.seed}`;
const bundleCache=new Map<string,Promise<SavedRoadster[]>>();
function loadBundle(url:string):Promise<SavedRoadster[]>{
  const cached=bundleCache.get(url);if(cached)return cached;
  const pending=fetch(url).then(async response=>{
    if(!response.ok)throw Error('Training recordings could not load.');
    const stream=new Blob([await response.arrayBuffer()]).stream().pipeThrough(new DecompressionStream('gzip'));
    const payload=await new Response(stream).json();return payload.flights as SavedRoadster[];
  });
  bundleCache.set(url,pending);pending.catch(()=>bundleCache.delete(url));
  while(bundleCache.size>3)bundleCache.delete(bundleCache.keys().next().value!);
  return pending;
}
function wilson(rate:number,n:number):string{
  if(!n||!Number.isFinite(rate))return 'Not available';
  const z=1.96, d=1+z*z/n, centre=(rate+z*z/(2*n))/d;
  const margin=z*Math.sqrt((rate*(1-rate)+z*z/(4*n))/n)/d;
  return `${percent(Math.max(0,centre-margin))}–${percent(Math.min(1,centre+margin))}`;
}

export function Saved3DResults(){
  const [index,setIndex]=useState<Index|null>(null),[runKey,setRunKey]=useState(''),[source,setSource]=useState<'evaluation'|'training'>('evaluation');
  const [episode,setEpisode]=useState(''),[flight,setFlight]=useState<SavedRoadster|null>(null),[error,setError]=useState(''),[loading,setLoading]=useState(true);
  useEffect(()=>{let active=true;fetch('/data/roadster3d/index.json').then(r=>{if(!r.ok)throw Error('No 3D results have been exported yet.');return r.json()})
    .then((data:Index)=>{if(active){setIndex(data);if(data.runs[0])setRunKey(keyOf(data.runs[0]))}})
    .catch(e=>{if(active)setError(e.message)}).finally(()=>{if(active)setLoading(false)});return()=>{active=false}},[]);
  const run=index?.runs.find(item=>keyOf(item)===runKey);
  const attempts=source==='training'?(run?.training_flights??[]):(run?.evaluations??[]);
  const selected=attempts.find(item=>String(item.episode)===episode)??attempts[0];
  useEffect(()=>{let active=true;setFlight(null);setError('');if(!selected)return()=>{active=false};
    const load=selected.bundle_url?loadBundle(selected.bundle_url).then(items=>items[selected.bundle_index??0]):
      fetch(selected.flight_url??'').then(r=>{if(!r.ok)throw Error('3D trajectory could not load.');return r.json()});
    load.then((data:SavedRoadster)=>{if(!data)throw Error('Recorded episode is missing from its bundle.');if(active)setFlight(data)}).catch(e=>{if(active)setError(e.message)});
    return()=>{active=false};},[selected?.flight_url,selected?.bundle_url,selected?.bundle_index]);
  const replay=useMemo(()=>flight?roadsterFlightFigure(flight):null,[flight]);
  const curve=useMemo(()=>run?.training_curve.filter(p=>p.timesteps!=null&&p.return!=null)??[],[run]);
  const trainingTrace=useMemo(()=>[trace(curve.map(p=>p.timesteps),curve.map(p=>p.return as number),'PPO training return','#eeeeee')],[curve]);
  const gasTrace=useMemo(()=>{
    if(!flight||!flight.columns.includes('gas_kg'))return [];
    const steps=flight.rows.map(row=>row[flight.columns.indexOf('step')]);
    return [trace(steps,flight.rows.map(row=>row[flight.columns.indexOf('gas_kg')]),'Gas remaining (kg)','#eeeeee')];
  },[flight]);
  const thrustTrace=useMemo(()=>{
    if(!flight||!flight.columns.includes('actual_lift_n'))return [];
    const steps=flight.rows.map(row=>row[flight.columns.indexOf('step')]);
    return [trace(steps,flight.rows.map(row=>row[flight.columns.indexOf('actual_lift_n')]/1000),'Actual lift (kN)','#eeeeee')];
  },[flight]);
  if(loading)return <div className="chart-empty">Loading saved 3D runs…</div>;
  if(!index?.runs.length)return <div className="roadster-note"><span>{error||'No evaluated Python 3D runs have been exported yet. Train and evaluate a model, then run the 3D exporter.'}</span></div>;
  return <section className="saved-3d-results">
    <div className="roadster-note"><span><strong>Recorded 3D results.</strong> These trajectories come from the Python Rapier simulator. {run?.policy_type==='rule_based'?'This hand-written reference uses privileged simulator state and contact; it is not a neural network.':'This run uses an evaluated PPO checkpoint.'} {run?.stage?.startsWith('flight')?'This run starts above the landing pad; it does not include a ramp launch.':'This run starts at the ramp.'} They use an experimental vehicle proxy. Small runs are pilots and do not establish a reliable landing policy.</span></div>
    {error&&<div className="notice" role="alert">{error}</div>}
    <div className="roadster-toolbar"><div><span className="toolbar-kicker">PYTHON / RAPIER 3D</span><h2>Explore recorded attempts.</h2></div>
      <div className="policy-tools"><label className="policy-select"><span>RUN</span><select value={runKey} onChange={e=>{setRunKey(e.target.value);setEpisode('')}}>{index.runs.map(r=><option value={keyOf(r)} key={keyOf(r)}>v{r.simulator_version} · {r.stage?.startsWith('flight')?'flight curriculum':'full task'} · {r.label} · seed {r.seed} · {r.policy_type==='ppo'?`${number(r.timesteps,0)} steps`:'reference'}</option>)}</select></label>
        <label className="policy-select"><span>SOURCE</span><select value={source} onChange={e=>{setSource(e.target.value as 'evaluation'|'training');setEpisode('')}}><option value="evaluation">Held-out evaluation</option><option value="training">Training episodes</option></select></label>
        <label className="policy-select"><span>ATTEMPT</span><select value={String(selected?.episode??'')} disabled={!attempts.length} onChange={e=>setEpisode(e.target.value)}>{attempts.map(a=><option key={a.episode} value={a.episode}>#{a.episode} · {a.outcome}</option>)}{!attempts.length&&<option value="">No attempts</option>}</select></label>
      </div></div>
    {run&&<div className="roadster-metrics">{[{label:'Evaluation episodes',value:number(run.summary.episodes,0)},{label:'Safe landing rate',value:percent(run.summary.safe_landing_rate)},{label:'Crash / launch failure rate',value:percent(run.summary.crash_rate)},{label:'Mean task return',value:number(run.summary.mean_native_return)}].map(item=><div className="roadster-metric surface" key={item.label}><strong>{item.value}</strong><span>{item.label}</span></div>)}</div>}
    <section className="surface roadster-stage"><div className="panel-heading"><div><span className="eyebrow">RECORDED RIGID-BODY TRAJECTORY</span><h2>{selected?.outcome??'No attempt selected'}</h2></div></div>
      {replay?<Plot label="Recorded 3D Python simulator trajectory" data={replay.data} customLayout={replay.layout} frames={replay.frames} className="roadster-plot"/>:<div className="chart-empty">{selected?'Loading recorded trajectory…':'No recordings for this source.'}</div>}
    </section>
    {flight?.simulator_version===7&&<div className="two-columns"><section className="surface"><span className="eyebrow">TANK</span><h2>Gas remaining</h2><Plot label="Recorded tank gas mass" data={gasTrace} customLayout={{xaxis:{gridcolor:'#222222',title:{text:'Episode step'}}}}/></section><section className="surface"><span className="eyebrow">NOZZLES</span><h2>Actual lift thrust</h2><Plot label="Recorded physical lift thrust" data={thrustTrace} customLayout={{xaxis:{gridcolor:'#222222',title:{text:'Episode step'}}}}/></section></div>}
    <div className="two-columns"><section className="surface"><span className="eyebrow">PPO TRAINING</span><h2>Learning curve</h2><p className="muted small">Training reward for this seed; held-out evaluation is reported above.</p>{curve.length?<Plot label="Recorded PPO training return" data={trainingTrace}/>:<div className="chart-empty">No completed episode returns were logged.</div>}</section>
      <section className="surface"><span className="eyebrow">RUN DETAILS</span><h2>Recorded evidence</h2><p className="muted small">{number(run?.timesteps,0)} training steps · {run?.evaluations.length??0} held-out episodes · {run?.training_flights.length??0} recorded training attempts. Select any attempt above to replay its actual 3D states.</p>{run&&<p className="muted small">Estimated landing rate 95% Wilson interval: {wilson(run.summary.safe_landing_rate,run.summary.episodes)}. This describes variation across evaluation scenarios for this checkpoint, not uncertainty across independently trained policies. Training budgets and simulator stages differ between runs.</p>}</section></div>
  </section>;
}
