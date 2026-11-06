import {initializeRapier,trainBrowserPPO,type RewardVariant} from './browser-ppo';

type StartMessage={type:'start';variant:RewardVariant;seed:number;episodes:number};
let stopRequested=false;
self.onmessage=async(event:MessageEvent<StartMessage|{type:'stop'}>)=>{
  if(event.data.type==='stop'){stopRequested=true;return;}
  stopRequested=false;
  try{
    self.postMessage({type:'status',message:'Loading Rapier3D WebAssembly…'});
    await initializeRapier();
    if(stopRequested){self.postMessage({type:'stopped',policy:null});return;}
    const result=await trainBrowserPPO({variant:event.data.variant,seed:event.data.seed,episodes:event.data.episodes,
      shouldStop:()=>stopRequested,
      onEpisode:(episode,updates)=>self.postMessage({type:'episode',episode,updates}),
    });
    self.postMessage({type:result.stopped?'stopped':'complete',policy:result.policy});
  }catch(error){self.postMessage({type:'error',message:error instanceof Error?error.message:String(error)});}
};
