import {useEffect,useMemo,useState} from 'react';
import {IconActivity,IconAlertTriangle,IconCar,IconCheck,IconClock,IconFlask,IconGauge,IconRoute,IconTarget,IconWind}
  from '@tabler/icons-react';
import {Plot,trace} from '@/components/Plot';
import {BrowserTraining} from '@/components/BrowserTraining';
import {Saved3DResults} from '@/components/Saved3DResults';
import {number,percent,type Row} from '@/lib/api';
import {roadsterFlightFigure,type SavedRoadster} from '@/lib/roadster-figure';

type Attempt={episode:number;eval_seed?:number;outcome:string;flight_url?:string;bundle_index?:number;native_return:number;shaped_return:number;
  safe_landing?:boolean;crash?:boolean;truncated?:boolean;landing_x_error?:number;touchdown_speed?:number;gas_remaining?:number};
type CurvePoint={timesteps:number;mean_episode_return:number|null;mean_episode_length:number|null};
type SeedCurve={variant:string;label:string;seed:number;rows:CurvePoint[]};
type FailureSource='training'|'evaluation';
type FailureRow={variant:string;label:string;seed:number;source:FailureSource;attempt:Attempt};
type Run={variant:string;label:string;seed:number;summary:Row;evaluations:Attempt[];training_flights:Attempt[];training_bundle_url?:string|null;timesteps_completed?:number;training_episodes?:number;training_curve_url?:string;behavior_cloning?:{episodes?:number;dagger_rounds?:number;training_transitions?:number;final_mse?:number;epoch_losses?:number[]}};
type Comparison={variant:string;variant_label:string;training_seeds:number;mean_safe_landing_rate:number;ci95_mean_safe_landing_rate_low:number|null;ci95_mean_safe_landing_rate_high:number|null;mean_native_return:number;std_native_return_across_seeds:number|null;ci95_mean_native_return_low:number|null;ci95_mean_native_return_high:number|null;mean_reward_gap:number};
type Index={schema:number;simulator_version:number;runs:Run[];comparison?:Comparison[];reference_evaluation?:{safe_landings:number;episodes:number;safe_landing_rate:number};preview:{label:string;flight_url:string;outcome:string;native_return:number};note:string};

const VARIANT_CURVE_STYLE:Record<string,{color:string;dash:'solid'|'dash'|'dot';fill:string;label:string}>={
  baseline:{color:'#f4f4f4',dash:'solid',fill:'rgba(244,244,244,0.10)',label:'Task reward'},
  potential_shaping:{color:'#aaaaaa',dash:'dash',fill:'rgba(170,170,170,0.10)',label:'Potential shaping'},
  upright_bonus:{color:'#d2d2d2',dash:'dot',fill:'rgba(210,210,210,0.08)',label:'Upright bonus'},
};
const FAILURE_PAGE_SIZE=25;
const T_95_BY_SAMPLE_SIZE:Record<number,number>={2:12.706,3:4.303,4:3.182,5:2.776};

function seedMeanTraces(curves:SeedCurve[],field:'mean_episode_return'|'mean_episode_length'):Row[]{
  const traces:Row[]=[];
  for(const [variant,style] of Object.entries(VARIANT_CURVE_STYLE)){
    const byStep=new Map<number,number[]>();
    for(const run of curves.filter(curve=>curve.variant===variant))for(const point of run.rows){
      const value=point[field];
      if(value==null||!Number.isFinite(value))continue;
      const values=byStep.get(point.timesteps)||[];values.push(value);byStep.set(point.timesteps,values);
    }
    const points=[...byStep.entries()].sort((a,b)=>a[0]-b[0]).filter(([,values])=>values.length>0);
    if(!points.length)continue;
    const x=points.map(([step])=>step);
    const mean=points.map(([,values])=>values.reduce((sum,value)=>sum+value,0)/values.length);
    const halfWidth=points.map(([,values],i)=>{
      if(values.length<2)return 0;
      const variance=values.reduce((sum,value)=>sum+(value-mean[i])**2,0)/(values.length-1);
      return (T_95_BY_SAMPLE_SIZE[values.length]??2.776)*Math.sqrt(variance/values.length);
    });
    if(points.some(([,values])=>values.length>1)){
      const low=trace(x,mean.map((value,i)=>value-halfWidth[i]),`${style.label} CI lower`,style.color,'solid',0);
      low.line={color:'rgba(0,0,0,0)',width:0};low.hoverinfo='skip';low.showlegend=false;low.legendgroup=variant;
      const high=trace(x,mean.map((value,i)=>value+halfWidth[i]),`${style.label} 95% CI`,style.color,'solid',0);
      high.line={color:'rgba(0,0,0,0)',width:0};high.fill='tonexty';high.fillcolor=style.fill;high.hoverinfo='skip';high.showlegend=false;high.legendgroup=variant;
      traces.push(low,high);
    }
    const meanTrace=trace(x,mean,`${style.label} · ${points[0][1].length} seeds`,style.color,style.dash,2.5);
    meanTrace.legendgroup=variant;traces.push(meanTrace);
  }
  return traces;
}

const trainingBundleCache=new Map<string,Promise<SavedRoadster[]>>();
function loadTrainingBundle(url:string):Promise<SavedRoadster[]>{
  const cached=trainingBundleCache.get(url);if(cached){trainingBundleCache.delete(url);trainingBundleCache.set(url,cached);return cached;}
  const pending=fetch(url).then(async response=>{
    if(!response.ok)throw Error('The saved training episodes could not load.');
    const compressed=await response.arrayBuffer();
    const decoded=new Blob([compressed]).stream().pipeThrough(new DecompressionStream('gzip'));
    const payload=await new Response(decoded).json();
    return payload.flights as SavedRoadster[];
  });
  trainingBundleCache.set(url,pending);pending.catch(()=>trainingBundleCache.delete(url));
  while(trainingBundleCache.size>2)trainingBundleCache.delete(trainingBundleCache.keys().next().value!);
  return pending;
}

export function RoadsterLab(){
  const [index,setIndex]=useState<Index|null>(null),[error,setError]=useState(''),[loading,setLoading]=useState(true);
  const [workspaceMode,setWorkspaceMode]=useState<'saved'|'saved3d'|'browser'>('saved3d');
  const [selected,setSelected]=useState(''),[mode,setMode]=useState<'evaluation'|'training'>('evaluation');
  const [episode,setEpisode]=useState(''),[saved,setSaved]=useState<SavedRoadster|null>(null),[flightLoading,setFlightLoading]=useState(false);
  const [trainingCurve,setTrainingCurve]=useState<CurvePoint[]>([]);
  const [allTrainingCurves,setAllTrainingCurves]=useState<SeedCurve[]>([]),[curvesLoading,setCurvesLoading]=useState(false);
  const [failureSource,setFailureSource]=useState<'all'|FailureSource>('all'),[failureOutcome,setFailureOutcome]=useState<'all'|'Crash'|'Timeout'>('all'),[failurePage,setFailurePage]=useState(0);
  useEffect(()=>{let active=true;fetch('/data/roadster/index.json').then(r=>{if(!r.ok)throw Error('Roadster results are not exported yet.');return r.json()}).then(d=>{if(active){setIndex(d);setSelected(d.runs[0]?`${d.runs[0].variant}:${d.runs[0].seed}`:'preview')}}).catch(e=>{if(active)setError(e.message)}).finally(()=>{if(active)setLoading(false)});return()=>{active=false}},[]);
  const run=index?.runs.find(r=>`${r.variant}:${r.seed}`===selected)||null;
  const preview=!run;
  useEffect(()=>{
    let active=true;
    if(!run?.training_curve_url){setTrainingCurve([]);return()=>{active=false}};
    fetch(run.training_curve_url).then(r=>{if(!r.ok)throw Error('The saved training curve could not load.');return r.json()})
      .then(d=>{if(active)setTrainingCurve(d.rows||[])})
      .catch(e=>{if(active)setError(e.message)});
    return()=>{active=false};
  },[run?.training_curve_url]);
  useEffect(()=>{
    if(!index)return;
    let active=true;setCurvesLoading(true);
    Promise.all(index.runs.filter(r=>r.training_curve_url).map(async run=>{
      const response=await fetch(run.training_curve_url!);
      if(!response.ok)throw Error(`Training curve could not load for ${run.label}, seed ${run.seed}.`);
      const data=await response.json();
      return {variant:run.variant,label:run.label,seed:run.seed,rows:(data.rows||[]) as CurvePoint[]};
    })).then(curves=>{if(active)setAllTrainingCurves(curves)}).catch(e=>{if(active)setError(e.message)}).finally(()=>{if(active)setCurvesLoading(false)});
    return()=>{active=false};
  },[index]);
  const attempts=useMemo(()=>run?(mode==='evaluation'?run.evaluations:run.training_flights):[],[run,mode]);
  const current=attempts.find(a=>String(a.episode)===episode)||attempts[0];
  const failedAttempts=useMemo<FailureRow[]>(()=>index?.runs.flatMap(savedRun=>[
    ...savedRun.training_flights.filter(a=>a.outcome!=='Safe landing').map(attempt=>({variant:savedRun.variant,label:savedRun.label,seed:savedRun.seed,source:'training' as const,attempt})),
    ...savedRun.evaluations.filter(a=>a.outcome!=='Safe landing').map(attempt=>({variant:savedRun.variant,label:savedRun.label,seed:savedRun.seed,source:'evaluation' as const,attempt})),
  ]).sort((a,b)=>a.variant.localeCompare(b.variant)||a.seed-b.seed||a.source.localeCompare(b.source)||a.attempt.episode-b.attempt.episode)||[],[index]);
  const filteredFailures=useMemo(()=>failedAttempts.filter(row=>(failureSource==='all'||row.source===failureSource)&&(failureOutcome==='all'||row.attempt.outcome===failureOutcome)),[failedAttempts,failureSource,failureOutcome]);
  const failurePages=Math.max(1,Math.ceil(filteredFailures.length/FAILURE_PAGE_SIZE));
  const visibleFailures=filteredFailures.slice(failurePage*FAILURE_PAGE_SIZE,(failurePage+1)*FAILURE_PAGE_SIZE);
  const failureCounts=useMemo(()=>({
    training:failedAttempts.filter(row=>row.source==='training').length,
    evaluation:failedAttempts.filter(row=>row.source==='evaluation').length,
    crashes:failedAttempts.filter(row=>row.attempt.outcome==='Crash').length,
    timeouts:failedAttempts.filter(row=>row.attempt.outcome==='Timeout').length,
  }),[failedAttempts]);
  const flightUrl=preview?index?.preview.flight_url:mode==='training'?run?.training_bundle_url:current?.flight_url;
  const metric=(key:string)=>run?.summary?.[key];
  const interval=(key:string,asPercent=false)=>{
    const low=metric(`ci95_${key}_low`),high=metric(`ci95_${key}_high`);
    if(low==null||high==null)return '95% CI unavailable';
    return `95% CI ${asPercent?percent(low):number(low)}–${asPercent?percent(high):number(high)}`;
  };
  useEffect(()=>{let active=true;if(!flightUrl){setSaved(null);return()=>{active=false}}setFlightLoading(true);
    const load=mode==='training'&&current?loadTrainingBundle(flightUrl).then(flights=>flights[current.bundle_index??0]):fetch(flightUrl).then(r=>{if(!r.ok)throw Error('The saved Roadster flight could not load.');return r.json()});
    load.then(d=>{if(active)setSaved(d)}).catch(e=>{if(active)setError(e.message)}).finally(()=>{if(active)setFlightLoading(false)});return()=>{active=false}},[flightUrl,mode,current?.bundle_index]);
  const replay=useMemo(()=>saved?roadsterFlightFigure(saved):null,[saved]);
  const rows=replay?.flight||[];
  const speed=useMemo(()=>rows.length?[trace(rows.map(r=>r.step),rows.map(r=>Math.hypot(Number(r.vx),Number(r.vz))),'Speed','#ededed'),trace(rows.map(r=>r.step),rows.map(r=>Number(r.gas)*20),'Gas × 20','#777777','dash')]:[],[rows]);
  const reward=useMemo(()=>rows.length?[trace(rows.map(r=>r.step),rows.map(r=>r.native_total),'Task return','#eeeeee'),trace(rows.map(r=>r.step),rows.map(r=>r.shaped_total),'Training return','#858585','dash')]:[],[rows]);
  const learningReturn=useMemo(()=>trainingCurve.filter(p=>p.mean_episode_return!=null).length?[trace(trainingCurve.filter(p=>p.mean_episode_return!=null).map(p=>p.timesteps),trainingCurve.filter(p=>p.mean_episode_return!=null).map(p=>p.mean_episode_return),'Mean training return','#eeeeee')]:[],[trainingCurve]);
  const learningLength=useMemo(()=>trainingCurve.filter(p=>p.mean_episode_length!=null).length?[trace(trainingCurve.filter(p=>p.mean_episode_length!=null).map(p=>p.timesteps),trainingCurve.filter(p=>p.mean_episode_length!=null).map(p=>p.mean_episode_length),'Mean episode length','#bbbbbb')]:[],[trainingCurve]);
  const cloningLoss=useMemo(()=>run?.behavior_cloning?.epoch_losses?.length?[trace(run.behavior_cloning.epoch_losses.map((_,i)=>i+1),run.behavior_cloning.epoch_losses,'Behavior-cloning loss','#d0d0d0')]:[],[run]);
  const allSeedReturn=useMemo(()=>seedMeanTraces(allTrainingCurves,'mean_episode_return'),[allTrainingCurves]);
  const allSeedLength=useMemo(()=>seedMeanTraces(allTrainingCurves,'mean_episode_length'),[allTrainingCurves]);
  const reviewFailure=(failure:FailureRow)=>{
    setSelected(`${failure.variant}:${failure.seed}`);setMode(failure.source);setEpisode(String(failure.attempt.episode));
    document.querySelector('.roadster-toolbar')?.scrollIntoView({behavior:'smooth',block:'start'});
  };
  return <div className="roadster-lab">
    <section className="roadster-hero">
      <div className="hero-grid"/><div className="roadster-hero-copy">
        <span className="eyebrow"><span className="tiny-square"/>ROADSTER FLIGHT LAB · PYTHON V7 / BROWSER V6</span>
        <h1>Ramp to touchdown.<br/><span>One policy at a time.</span></h1>
        <p>Train a policy to drive up the physical Tesla-style ramp, clear its curved shoulder, and steer back to a landing zone. Gravity, contact forces, thrust and failures are simulated in 3D.</p>
        <div className="roadster-hero-meta"><span><IconActivity size={14}/>PPO FROM SCRATCH</span><span><IconWind size={14}/>CURVED-TOP RAMP + SIX JETS</span><span><IconRoute size={14}/>RAPIER3D · GRAVITY · CONTACTS</span></div>
      </div>
      <div className="roadster-silhouette" aria-hidden="true">
        <svg className="roadster-ramp-art" viewBox="0 0 360 360" preserveAspectRatio="xMidYMid meet"><path d="M 196 354 C 263 354 290 324 290 260 L 290 126 C 290 94 278 64 255 42 L 210 4"/></svg>
        <div className="car-shape"><i/><b/><em/></div><div className="roadster-boost"/>
      </div>
    </section>
    <div className="experiment-mode-switch" role="group" aria-label="Choose experiment view">
      <button className={workspaceMode==='saved3d'?'active':''} aria-pressed={workspaceMode==='saved3d'} onClick={()=>setWorkspaceMode('saved3d')}><span className="mode-step">01</span><span><strong>Saved 3D experiment</strong><small>Python PPO · recorded Rapier trajectories</small></span><span className="mode-state">RESULTS</span></button>
      <button className={workspaceMode==='browser'?'active':''} aria-pressed={workspaceMode==='browser'} onClick={()=>setWorkspaceMode('browser')}><span className="mode-step">02</span><span><strong>Train v6 in your browser</strong><small>Rapier3D WebAssembly · curved-top Tesla ramp · local PPO</small></span><span className="mode-state">ON DEVICE</span></button>
      <button className={workspaceMode==='saved'?'active':''} aria-pressed={workspaceMode==='saved'} onClick={()=>setWorkspaceMode('saved')}><span className="mode-step">03</span><span><strong>Saved v3 experiment</strong><small>Python · planar legacy runs</small></span><span className="mode-state">ARCHIVE</span></button>
    </div>
    {workspaceMode==='saved3d'&&<Saved3DResults/>}
    <div className={`browser-training-shell ${workspaceMode==='browser'?'':'is-hidden'}`} aria-hidden={workspaceMode!=='browser'}><BrowserTraining/></div>
    {workspaceMode==='saved'&&<>
    <div className="roadster-note"><IconAlertTriangle size={16}/><span><strong>Archived v3 results:</strong> these saved Python runs use the earlier planar, scripted-ramp simulator; their 3D depth is illustrative. They are historical and not comparable to Rapier3D v6 training. The new ramp has a physical curved-top mesh collider; the vehicle remains a simplified rigid-body proxy, not Tesla hardware or a validated vehicle simulation. {index?.note} The archived non-ML PD reference landed safely on {index?.reference_evaluation?.safe_landings??'—'}/{index?.reference_evaluation?.episodes??'—'} shared launches.</span></div>
    {error&&<div className="notice" role="alert">{error}<button onClick={()=>setError('')} aria-label="Dismiss error">×</button></div>}
    <div className="roadster-toolbar">
      <div><span className="toolbar-kicker">FLIGHT WORKSPACE</span><h2>Inspect the maneuver.</h2></div>
      <div className="policy-tools">
        <label className="policy-select"><span>POLICY / REFERENCE</span><select aria-label="Roadster policy" value={selected} onChange={e=>{setSelected(e.target.value);setEpisode('')}} disabled={!index?.runs.length}>{index?.runs.map(r=><option key={`${r.variant}:${r.seed}`} value={`${r.variant}:${r.seed}`}>{r.label} · seed {r.seed}</option>)}<option value="preview">Rule-based reference flight</option></select></label>
        {run&&<label className="policy-select"><span>FLIGHT SOURCE</span><select value={mode} onChange={e=>setMode(e.target.value as 'evaluation'|'training')}><option value="evaluation">Evaluation episodes</option><option value="training" disabled={!run.training_flights.length}>Training examples</option></select></label>}
        {run&&<label className="policy-select"><span>ATTEMPT</span><select value={String(current?.episode??'')} onChange={e=>setEpisode(e.target.value)} disabled={!attempts.length}>{attempts.map(a=><option key={a.episode} value={a.episode}>#{a.episode} · {a.outcome}</option>)}{!attempts.length&&<option>No exported flights</option>}</select></label>}
      </div>
    </div>
    {!index?.runs.length&&!loading&&<div className="roadster-preview-banner"><IconFlask size={17}/><div><strong>No saved v3 policy runs are exported.</strong><span>The historical preview uses a reference controller. New 3D training runs use <code>python -m rocket_landing.cli roadster3d-train --variant potential_shaping --seed 0</code>.</span></div></div>}
    {!!run&&(run.evaluations.length<100||index!.runs.filter(r=>r.variant===run.variant).length<5)&&<div className="roadster-preview-banner roadster-pilot-banner"><IconFlask size={17}/><div><strong>Pilot results · {number(run.timesteps_completed)} training steps · {run.evaluations.length} held-out episodes</strong><span>This is an early run, not the five-seed, 100-episode comparison. Use the confidence intervals and failure replays below; do not describe this result as solved.</span></div></div>}
    {loading&&<div className="chart-empty">Loading saved Roadster results…</div>}
    {!!run&&<div className="roadster-metrics">
      {[{title:'Safe landings',value:percent(metric('safe_landing_rate')),detail:interval('safe_landing',true),icon:IconCheck},
        {title:'Mean touchdown speed',value:`${number(metric('mean_touchdown_speed'))} m/s`,detail:interval('touchdown_speed'),icon:IconGauge},
        {title:'Mean pad error',value:`${number(metric('mean_landing_x_error'))} m`,detail:interval('landing_x_error'),icon:IconTarget},
        {title:'Mean gas remaining',value:percent(metric('mean_gas_remaining')),detail:interval('gas_remaining',true),icon:IconWind}].map(m=><div className="roadster-metric surface" key={m.title}><m.icon size={16}/><strong>{m.value}</strong><span>{m.title}</span><small>{m.detail}</small></div>)}
    </div>}
    <div className="roadster-layout">
      <section className="surface roadster-controls"><div className="section-icon"><IconCar size={21}/></div><span className="eyebrow">CONTROL CONTRACT</span><h2>Three booster channels.</h2>
        <p>During flight, PPO fine-tunes normalized horizontal impulse, vertical impulse, and attitude torque from a common PD-controller warm start. Gas use follows commanded thrust. Every policy receives the same scripted ramp trajectory; fan and suction forces are not simulated.</p>
        <div className="roadster-action"><span className="mono">01</span><div><strong>Horizontal jet</strong><small>Brakes or corrects lateral drift</small></div></div>
        <div className="roadster-action"><span className="mono">02</span><div><strong>Vertical jet</strong><small>Controls climb and touchdown speed</small></div></div>
        <div className="roadster-action"><span className="mono">03</span><div><strong>Attitude jet</strong><small>Stabilizes the car for touchdown</small></div></div>
        <div className="roadster-phase"><span className="status-dot"/><span>Ramp fixture: scripted</span><span className="status-dot"/><span>Flight: policy-controlled</span></div>
      </section>
      <section className="surface roadster-stage"><div className="panel-heading"><div><span className="eyebrow">TRAJECTORY VIEWER</span><h2>{flightLoading?'Loading flight…':saved?.outcome||'Saved flight replay'}</h2></div><span className="pill"><IconCar size={13}/>3D REPLAY</span></div>
        {replay?<Plot key={flightUrl} label="Roadster ramp, thruster, and landing trajectory from the saved simulator states" data={replay.data} customLayout={replay.layout} frames={replay.frames} className="roadster-plot"/>:<div className="flight-empty"><div className="viewer-orbit"><IconCar size={24}/></div><h3>{flightLoading?'Preparing replay…':'No flight selected.'}</h3><p>A saved reference or PPO trajectory appears here.</p></div>}
        <div className="viewer-hints"><span><IconRoute size={13}/>Ramp run-up is scripted</span><span><IconActivity size={13}/>White plumes show commanded jets</span><span><IconClock size={13}/>Use timeline to replay</span></div>
      </section>
    </div>
    {!!index?.runs.length&&<>
      <div className="section-heading"><div><span className="eyebrow">ALL TRAINING SEEDS</span><h2>Learning curves across the experiment.</h2></div><span className="muted small">{allTrainingCurves.length} runs · 95% t intervals across seeds</span></div>
      <div className="two-columns roadster-all-curves">
        <section className="surface"><span className="eyebrow">PPO REWARD</span><h2>Training return by reward design</h2><p className="muted small">Each line averages independent seeds; the shaded band shows the 95% confidence interval.</p>{curvesLoading?<div className="chart-empty">Loading all training curves…</div>:allSeedReturn.length?<Plot label="Mean PPO training return across all reward designs and seeds, with 95 percent confidence intervals" data={allSeedReturn} customLayout={{showlegend:true,xaxis:{title:{text:'Training timesteps'},gridcolor:'#252525'},yaxis:{title:{text:'Mean episode return'},gridcolor:'#252525'}}}/>:<div className="chart-empty">No aggregate return curves were exported.</div>}</section>
        <section className="surface"><span className="eyebrow">EPISODE DYNAMICS</span><h2>Episode length by reward design</h2><p className="muted small">Longer episodes can indicate loitering or timeouts; compare with the return and failure records.</p>{curvesLoading?<div className="chart-empty">Loading all training curves…</div>:allSeedLength.length?<Plot label="Mean PPO episode length across all reward designs and seeds, with 95 percent confidence intervals" data={allSeedLength} customLayout={{showlegend:true,xaxis:{title:{text:'Training timesteps'},gridcolor:'#252525'},yaxis:{title:{text:'Mean episode length'},gridcolor:'#252525'}}}/>:<div className="chart-empty">No aggregate length curves were exported.</div>}</section>
      </div>
    </>}
    {!!run&&<><div className="section-heading"><div><span className="eyebrow">SELECTED RUN DETAIL</span><h2>What changed in this seed?</h2></div><span className="muted small">{run.behavior_cloning?.episodes??0} PD demos + {run.behavior_cloning?.dagger_rounds??0} DAgger rounds · {number(run.training_episodes,0)} PPO training episodes</span></div><div className="two-columns roadster-curves"><section className="surface"><span className="eyebrow">BEHAVIOR-CLONING WARM START</span><h2>Demonstration loss</h2><p className="muted small">Actor mean-action MSE over PD and DAgger labels, before PPO fine-tuning.</p>{cloningLoss.length?<Plot label="Behavior-cloning training loss over supervised epochs" data={cloningLoss} customLayout={{xaxis:{title:{text:'Epoch'},gridcolor:'#252525'}}}/>:<div className="chart-empty">No behavior-cloning loss was exported for this run.</div>}</section><section className="surface"><span className="eyebrow">PPO ROLLOUT</span><h2>Mean training return</h2><p className="muted small">Selected reward for this seed. Compare it with the original task return in the evaluation above.</p>{learningReturn.length?<Plot label="Mean training episode return over PPO timesteps" data={learningReturn} customLayout={{xaxis:{title:{text:'Training timesteps'},gridcolor:'#252525'}}}/>:<div className="chart-empty">No training return curve was exported for this run.</div>}</section><section className="surface"><span className="eyebrow">EPISODE DYNAMICS</span><h2>Mean episode length</h2><p className="muted small">This run’s average episode length at each PPO update.</p>{learningLength.length?<Plot label="Mean episode length over training timesteps" data={learningLength} customLayout={{xaxis:{title:{text:'Training timesteps'},gridcolor:'#252525'}}}/>:<div className="chart-empty">No episode-length curve was exported for this run.</div>}</section></div></>}
    {!!index?.runs.length&&<>
      <div className="section-heading"><div><span className="eyebrow">FAILURE REVIEW</span><h2>Every crash and timeout.</h2></div><span className="muted small">{number(failedAttempts.length,0)} failed training and evaluation attempts</span></div>
      <section className="surface failure-panel">
        <p className="failure-intro">Choose any row to load its saved 3D trajectory, telemetry, and task-versus-training reward trace above.</p>
        <div className="failure-stats">
          {[{label:'All failures',value:failedAttempts.length},{label:'Training failures',value:failureCounts.training},{label:'Evaluation failures',value:failureCounts.evaluation},{label:'Crashes',value:failureCounts.crashes},{label:'Timeouts',value:failureCounts.timeouts}].map(item=><div className="failure-stat" key={item.label}><strong>{number(item.value,0)}</strong><span>{item.label}</span></div>)}
        </div>
        <div className="failure-toolbar">
          <div><span className="eyebrow">SAVED FAILURE RECORDS</span><p className="muted small">Training records include every completed PPO episode; evaluation records use the held-out launch set.</p></div>
          <div className="failure-filters">
            <label><span>DATA SOURCE</span><select value={failureSource} onChange={e=>{setFailureSource(e.target.value as 'all'|FailureSource);setFailurePage(0)}}><option value="all">Training + evaluation</option><option value="training">Training episodes</option><option value="evaluation">Evaluation episodes</option></select></label>
            <label><span>OUTCOME</span><select value={failureOutcome} onChange={e=>{setFailureOutcome(e.target.value as 'all'|'Crash'|'Timeout');setFailurePage(0)}}><option value="all">Crash + timeout</option><option value="Crash">Crash</option><option value="Timeout">Timeout</option></select></label>
          </div>
        </div>
        <div className="table-scroll failure-table-wrap"><table><thead><tr><th>Source</th><th>Reward design</th><th>Seed</th><th>Episode</th><th>Outcome</th><th>Task return</th><th>Training return</th><th></th></tr></thead><tbody>
          {visibleFailures.map(row=><tr key={`${row.source}:${row.variant}:${row.seed}:${row.attempt.episode}`}><td>{row.source==='training'?'PPO training':'Held-out evaluation'}</td><td>{row.label}</td><td className="mono">{row.seed}</td><td className="mono">{row.attempt.episode}</td><td><span className="failure-outcome">{row.attempt.outcome}</span></td><td className="mono">{number(row.attempt.native_return)}</td><td className="mono">{number(row.attempt.shaped_return)}</td><td><button className="review-attempt" onClick={()=>reviewFailure(row)}>Open replay ↗</button></td></tr>)}
          {!visibleFailures.length&&<tr><td colSpan={8} className="table-empty">No failed attempts match these filters.</td></tr>}
        </tbody></table></div>
        <div className="failure-pagination"><span className="muted small">{filteredFailures.length?`${failurePage*FAILURE_PAGE_SIZE+1}–${Math.min((failurePage+1)*FAILURE_PAGE_SIZE,filteredFailures.length)} of ${number(filteredFailures.length,0)}`:'0 results'}</span><div><button className="button ghost" disabled={failurePage===0} onClick={()=>setFailurePage(page=>Math.max(0,page-1))}>Previous</button><span className="mono">{failurePage+1} / {failurePages}</span><button className="button ghost" disabled={failurePage+1>=failurePages} onClick={()=>setFailurePage(page=>Math.min(failurePages-1,page+1))}>Next</button></div></div>
      </section>
    </>}
    <p className="fine-print">The saved experiment view displays the exported Python/SB3 results and trajectories. It does not run that training job on this site.</p>
    {saved&&<div className="two-columns"><section className="surface"><span className="eyebrow">THRUSTER ACCOUNTING</span><h2>Task return vs. training return.</h2><Plot label="Task reward and shaped training reward by simulator step" data={reward} customLayout={{showlegend:true,xaxis:{title:{text:'Simulator step'},gridcolor:'#252525'}}}/></section><section className="surface"><span className="eyebrow">FLIGHT TELEMETRY</span><h2>Speed and remaining gas.</h2><Plot label="Vehicle speed and remaining gas over the maneuver" data={speed} customLayout={{showlegend:true,xaxis:{title:{text:'Simulator step'},gridcolor:'#252525'}}}/></section></div>}
    {!!index?.runs.length&&<><div className="section-heading"><div><span className="eyebrow">PAIRED EVALUATION</span><h2>Compare reward designs.</h2></div><span className="muted small">Shared held-out launch conditions</span></div>
      {!!index.comparison?.length&&<div className="surface comparison-panel roadster-aggregate"><div className="table-scroll"><table><thead><tr><th>Reward design</th><th>Training seeds</th><th>Safe landing rate</th><th>Native return</th><th>Reward gap</th></tr></thead><tbody>{index.comparison.map(row=><tr key={row.variant}><td>{row.variant_label}</td><td className="mono">{number(row.training_seeds,0)}</td><td className="mono">{percent(row.mean_safe_landing_rate)}<small>{percent(row.ci95_mean_safe_landing_rate_low)}–{percent(row.ci95_mean_safe_landing_rate_high)} CI</small></td><td className="mono">{number(row.mean_native_return)} ± {number(row.std_native_return_across_seeds)}<small>{number(row.ci95_mean_native_return_low)}–{number(row.ci95_mean_native_return_high)} CI</small></td><td className="mono">{number(row.mean_reward_gap)}</td></tr>)}</tbody></table></div><p className="table-footnote">Aggregate confidence intervals summarize variation across independent training seeds; per-run intervals use held-out episodes.</p></div>}
      <div className="surface comparison-panel"><span className="eyebrow">PER-SEED RESULTS</span><div className="table-scroll"><table><thead><tr><th>Reward design</th><th>Seed</th><th>Episodes</th><th>Safe landings</th><th>Task return</th><th>Reward gap</th></tr></thead><tbody>{index.runs.map(r=><tr key={`${r.variant}:${r.seed}`} className={run&&r.variant===run.variant&&r.seed===run.seed?'active-row':''}><td><button className="table-policy" onClick={()=>setSelected(`${r.variant}:${r.seed}`)}><span className="table-dot"/>{r.label}</button></td><td className="mono">{r.seed}</td><td className="mono">{r.evaluations.length}</td><td className="mono">{percent(r.summary.safe_landing_rate)}</td><td className="mono">{number(r.summary.mean_native_return)} ± {number(r.summary.std_native_return)}</td><td className="mono">{number(r.summary.mean_reward_gap)}</td></tr>)}</tbody></table></div><p className="table-footnote">Rows are separate training seeds. Use safe landing rate and touchdown telemetry alongside shaped return.</p></div></>}
    </>}
  </div>;
}
