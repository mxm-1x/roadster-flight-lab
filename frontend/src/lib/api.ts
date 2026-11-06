import {savedFlightReplay,type SavedFlight} from './flight-figure';
export type Row = Record<string, any>;
export type Run = {variant: string; label: string; seed: number; has_model: boolean; status?: string};
export type RunDetail = Omit<Run,'status'> & {status: Row; summary: Row; training: Row[]; progress: Row[]; evaluation: Row[]; checkpoints: number[]; checkpoint_seeds:number[]; sessions: {id:string; episodes: {episode:number; outcome:string}[]}[]};
export type Replay = {figure: Row; flight: Row[]; outcome:string; native_return:number; shaped_return:number; steps:number};
export async function getJSON<T>(path:string):Promise<T> {
 const response=await fetch(path);
 if(!response.ok)throw Error('This saved result is not available. Please refresh and try again.');
 if(!response.headers.get('content-type')?.includes('application/json'))throw Error('Saved experiment data could not load.');
 return response.json();
}
export function getRuns(){return getJSON<{runs:Run[]}>('/data/runs.json')}
export function getRun(variant:string,seed:number){return getJSON<RunDetail>(`/data/runs/${variant}/${seed}.json`)}
export async function getReplay(payload:Row):Promise<Replay>{
 const base=`/data/flights/${payload.variant}/${payload.seed}`;
 const file=payload.mode==='evaluation'?`evaluation/${payload.episode}.json`:payload.mode==='checkpoint'?`checkpoint/${payload.checkpoint}_${payload.episode_seed}.json`:`recorded/${payload.session}/${payload.episode}.json`;
 return savedFlightReplay(await getJSON<SavedFlight>(`${base}/${file}`));
}
export const labels:Record<string,string> = {baseline:'Original reward', flawed_bonus:'Upright bonus', potential_shaping:'Progress shaping'};
export const keyOf = (run:Pick<Run,'variant'|'seed'>) => `${run.variant}:${run.seed}`;
export const number = (value:unknown, digits=1) => value === undefined || value === null || !Number.isFinite(Number(value)) ? '—' : Number(value).toLocaleString(undefined,{maximumFractionDigits:digits,minimumFractionDigits:digits});
export const percent = (value:unknown) => value === undefined || value === null ? '—' : `${number(Number(value)*100,0)}%`;
