import {useEffect, useRef} from 'react';
import type {Row} from '@/lib/api';
let plotlyPromise:Promise<any> | undefined;
function getPlotly() {
  if (!plotlyPromise) plotlyPromise = new Promise((resolve,reject) => {
    const script = document.createElement('script'); script.src='/plotly.min.js';
    script.onload=()=>resolve((window as any).Plotly); script.onerror=()=>{plotlyPromise=undefined; reject(Error('The chart library could not load. Refresh to try again.'));};
    document.head.appendChild(script);
  });
  return plotlyPromise;
}
const layout = {paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',font:{family:'Inter, Arial, sans-serif',color:'#929292',size:11},margin:{l:45,r:16,t:16,b:40},xaxis:{gridcolor:'#222222',zeroline:false,title:{text:'Training timesteps',font:{size:10,color:'#777777'}}},yaxis:{gridcolor:'#222222',zeroline:false},showlegend:false,hovermode:'x unified',hoverlabel:{bgcolor:'#1c1c1c',bordercolor:'#444444',font:{color:'#ffffff'}},legend:{orientation:'h',x:0,y:1.15},colorway:['#ffffff','#888888','#555555']};
export function trace(x:unknown[],y:unknown[],name:string,color='#ffffff',dash='solid',width=2):Row {return {x,y,name,type:'scatter',mode:'lines',line:{color,width,dash},hovertemplate:'%{y:.2f}<extra>'+name+'</extra>'};}
export function Plot({data,customLayout,frames,className='',label}:{data:Row[];customLayout?:Row;frames?:Row[];className?:string;label:string}) {
  const node = useRef<HTMLDivElement>(null);
  const customLayoutKey=JSON.stringify(customLayout??{});
  useEffect(()=>{
    let alive=true; let resize:ResizeObserver|undefined;
    getPlotly().then(async P=>{
      if (!alive || !node.current) return;
      const el=node.current;
      const effectiveLayout:Row={...layout,...customLayout};
      if(frames){delete effectiveLayout.height;delete effectiveLayout.width;effectiveLayout.autosize=true;}
      const config={responsive:true,displaylogo:false,displayModeBar:!!frames,modeBarButtonsToRemove:['toImage']};
      if(frames?.length) await P.react(el,{data,layout:effectiveLayout,frames,config});
      else await P.react(el,data,effectiveLayout,config);
      if (!alive) return;
      resize=new ResizeObserver(()=>{ if (el.offsetWidth && el.offsetHeight) P.Plots.resize(el); }); resize.observe(el);
    }).catch(error=>{console.error('[Landing Lab] Plotly render failed',error);if(alive&&node.current)node.current.textContent=error.message;});
    return ()=>{alive=false;resize?.disconnect();};
  },[data,customLayoutKey,frames]);
  useEffect(()=>{const element=node.current;return()=>{if(element && (window as any).Plotly)(window as any).Plotly.purge(element);};},[]);
  return <div ref={node} role="img" aria-label={label} className={`plot ${className}`} />;
}
