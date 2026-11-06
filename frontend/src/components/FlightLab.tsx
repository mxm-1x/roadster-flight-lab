import {useEffect,useMemo,useState,useRef} from 'react';
import {IconArrowUpRight,IconPlayerPlay,IconCube,IconLoader2,IconRoute,IconPointer,IconArrowsMove} from '@tabler/icons-react';
import {getReplay,number,type RunDetail,type Replay} from '@/lib/api';
import {Plot,trace} from './Plot';
export function FlightLab({detail,onError,initialEpisode}:{detail:RunDetail|null;onError:(s:string)=>void;initialEpisode:number|null}) {
  const modes = useMemo(()=>[...(detail?.evaluation.length?['evaluation']:[]),...(detail?.sessions.length?['recorded']:[]),...(detail?.checkpoints.length?['checkpoint']:[])],[detail]);
  const [mode,setMode]=useState('evaluation'),[outcome,setOutcome]=useState('All'),[episode,setEpisode]=useState(''),[session,setSession]=useState(''),[checkpoint,setCheckpoint]=useState(''),[seed,setSeed]=useState('10000');
  const [replay,setReplay]=useState<Replay|null>(null),[loading,setLoading]=useState(false);
  const effectiveMode=modes.includes(mode)?mode:modes[0]||'';
  const effectiveSession=detail?.sessions.find(s=>s.id===session)||detail?.sessions[0];
  const attempts=useMemo(()=>((effectiveMode==='evaluation'?detail?.evaluation:effectiveSession?.episodes)||[]).filter(r=>outcome==='All'||r.outcome===outcome),[effectiveMode,detail,effectiveSession,outcome]);
  const effectiveEpisode=attempts.some(r=>String(r.episode)===episode)?episode:String(attempts[0]?.episode??'');
  const effectiveCheckpoint=detail?.checkpoints.includes(Number(checkpoint))?checkpoint:String(detail?.checkpoints.at(-1)??'');
  const replayRequest=useRef(0);
  useEffect(()=>{replayRequest.current++;setReplay(null);setLoading(false);setEpisode(initialEpisode===null?'':String(initialEpisode));setOutcome('All');},[detail?.variant,detail?.seed,initialEpisode]);
  async function load() {
    if(!detail)return;const request=++replayRequest.current;setLoading(true);onError('');
    try {
      const payload:any={variant:detail.variant,seed:detail.seed,mode:effectiveMode};
      if(effectiveMode==='checkpoint'){payload.checkpoint=Number(effectiveCheckpoint);payload.episode_seed=Number(seed)}
      else {payload.episode=Number(effectiveEpisode);if(effectiveMode==='recorded')payload.session=effectiveSession?.id;}
      const result=await getReplay(payload);if(request===replayRequest.current)setReplay(result);
    } catch(e){if(request===replayRequest.current)onError((e as Error).message)} finally{if(request===replayRequest.current)setLoading(false)}
  }
  const reward=useMemo(()=>replay?[trace(replay.flight.map((_,i)=>i),replay.flight.map(r=>r.native_total),'Environment'),trace(replay.flight.map((_,i)=>i),replay.flight.map(r=>r.shaped_total),'Shaped','#888888','dash')]:[],[replay]);
  const velocity=useMemo(()=>replay?[trace(replay.flight.map((_,i)=>i),replay.flight.map(r=>r.vx),'Horizontal'),trace(replay.flight.map((_,i)=>i),replay.flight.map(r=>r.vy),'Vertical','#888888','dash')]:[],[replay]);
  const stepsLayout=useMemo(()=>({showlegend:true,xaxis:{title:{text:'Simulator step'},gridcolor:'#222222',zeroline:false}}),[]);
  return <>
    <div className="flight-layout">
      <section className="surface flight-controls">
        <div className="section-icon"><IconRoute size={20}/></div><h2>Flight configuration</h2><p className="muted">Every attempt has a story. Pick one to see how the policy behaved.</p>
        <label>Flight source<select value={effectiveMode} onChange={e=>{setMode(e.target.value);setEpisode('');setOutcome('All')}} disabled={!modes.length}>{modes.map(m=><option key={m} value={m}>{m==='evaluation'?'Evaluation flight':m==='recorded'?'Recorded training flight':'Checkpoint replay'}</option>)}{!modes.length&&<option>No flights available</option>}</select></label>
        {effectiveMode==='recorded'&&<label>Recording session<select value={effectiveSession?.id||''} onChange={e=>setSession(e.target.value)}>{detail?.sessions.map(s=><option key={s.id}>{s.id}</option>)}</select></label>}
        {effectiveMode==='checkpoint'?<><label>Training checkpoint<select value={effectiveCheckpoint} onChange={e=>setCheckpoint(e.target.value)}>{detail?.checkpoints.map(n=><option value={n} key={n}>{number(n,0)} steps</option>)}</select></label><label>Saved simulator seed<select value={seed} onChange={e=>setSeed(e.target.value)}>{(detail?.checkpoint_seeds||[10000]).map(n=><option key={n} value={n}>{n}</option>)}</select></label></>:<><label>Outcome<select value={outcome} onChange={e=>{setOutcome(e.target.value);setEpisode('')}}>{['All','Safe landing','Crash','Timeout','Unknown'].map(o=><option key={o}>{o}</option>)}</select></label><label>Attempt<select value={effectiveEpisode} onChange={e=>setEpisode(e.target.value)}>{attempts.map(a=><option key={a.episode} value={a.episode}>Attempt {a.episode} · {a.outcome}</option>)}{!attempts.length&&<option>No matching attempts</option>}</select></label></>}
        <button className="button primary" onClick={load} disabled={loading||!effectiveMode||(effectiveMode!=='checkpoint'&&!attempts.length)||(effectiveMode==='checkpoint'&&(!seed||!Number.isInteger(Number(seed))||Number(seed)<0||Number(seed)>2147483647))}>{loading?<><IconLoader2 className="spin" size={16}/>Replaying flight…</>:<>Load flight<IconArrowUpRight size={16}/></>}</button>
        <p className="fine-print">{effectiveMode==='recorded'?'States recorded during the actual training attempt.':'A saved deterministic flight from this policy and simulator seed.'}</p>
      </section>
      <section className="surface flight-stage"><div className="panel-heading"><div><span className="eyebrow">TRAJECTORY VIEWER</span><h2>The policy, in motion.</h2></div><span className="pill"><IconCube size={13}/>3D REPLAY</span></div>
        {replay?<Plot key={`${detail?.variant}:${detail?.seed}:${replay.steps}:${replay.native_return}`} label="Interactive 3D replay of the actual simulator trajectory" data={replay.figure.data} customLayout={replay.figure.layout} frames={replay.figure.frames} className="flight-plot"/>:<div className="flight-empty"><div className="viewer-orbit"><IconPlayerPlay size={24}/></div><h3>Ready for playback.</h3><p>Load a flight to explore its trajectory,<br/>engine decisions, and outcome.</p><span className="mono">AWAITING FLIGHT DATA</span></div>}
        <div className="viewer-hints"><span><IconArrowsMove size={13}/>Drag to orbit</span><span><IconPointer size={13}/>Scroll to zoom</span><span><IconPlayerPlay size={13}/>Use the timeline to play</span></div>
      </section>
    </div>
    <p className="fine-print">Actual 2D simulator position, tilt, and engine actions, presented in 3D. Depth is illustrative.</p>
    {replay&&<><div className="flight-stats">{[['Outcome',replay.outcome],['Simulator steps',number(replay.steps,0)],['Environment return',number(replay.native_return)],['Shaped return',number(replay.shaped_return)]].map(([k,v])=><div key={k}><span>{k}</span><strong>{v}</strong></div>)}</div><div className="two-columns"><section className="surface"><span className="eyebrow">REWARD ACCOUNTING</span><h2>What got rewarded?</h2><Plot label="Environment and shaped return through the flight" data={reward} customLayout={stepsLayout}/></section><section className="surface"><span className="eyebrow">FLIGHT TELEMETRY</span><h2>Velocity through the flight</h2><Plot label="Horizontal and vertical velocity" data={velocity} customLayout={stepsLayout}/></section></div></>}
  </>;
}
