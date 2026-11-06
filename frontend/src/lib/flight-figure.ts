import type {Replay,Row} from './api';

export type SavedFlight = {
 schema:number;columns:string[];rows:number[][];pad:[number,number];
 world_scale:[number,number,number];outcome:string;native_return:number;
 shaped_return:number;steps:number;
};

type Point3=[number,number,number];
type LinePoint=Point3|null;
type Face=[number,number,number];
const SEGMENTS=16;
const MODEL_RADIAL_SCALE=1.8;
const MODEL_HEIGHT_SCALE=1.8;
const MODEL_FOOT_LEVEL=-.52;
const METAL={ambient:.48,diffuse:.78,specular:.68,roughness:.3,fresnel:.15};

function movePoint([px,py,pz]:Point3,x:number,z:number,angle:number,scaleForModel=true):Point3 {
 const c=Math.cos(angle),s=Math.sin(angle);
 // Enlarge the small simulator lander for a clear silhouette. Stretch upward
 // from the foot plane so its landing gear still meets the same surface.
 const hullX=scaleForModel?px*MODEL_RADIAL_SCALE:px;
 const hullY=scaleForModel?py*MODEL_RADIAL_SCALE:py;
 const hullZ=scaleForModel?MODEL_FOOT_LEVEL+(pz-MODEL_FOOT_LEVEL)*MODEL_HEIGHT_SCALE:pz;
 return [x+hullX*c-hullZ*s,hullY,z+hullX*s+hullZ*c];
}

function mesh(vertices:Point3[],faces:Face[],color:string,name:string,opacity=1):Row {
 return {type:'mesh3d',x:vertices.map(([x])=>x),y:vertices.map(([,y])=>y),z:vertices.map(([, ,z])=>z),
  i:faces.map(([i])=>i),j:faces.map(([,j])=>j),k:faces.map(([, ,k])=>k),
  color,opacity,name,flatshading:false,lighting:METAL,showlegend:false};
}

function ringsTrace(
 rings:Array<{z:number;radius:number}>,color:string,name:string,x:number,z:number,angle:number,
 segments=SEGMENTS,opacity=1,
):Row {
 const vertices:Point3[]=rings.flatMap(({z:height,radius})=>Array.from({length:segments},(_,n)=>{
  const theta=2*Math.PI*n/segments;
  return movePoint([radius*Math.cos(theta),radius*Math.sin(theta),height],x,z,angle);
 }));
 const faces:Face[]=[];
 for(let r=0;r<rings.length-1;r++) for(let n=0;n<segments;n++){
  const a=r*segments+n,b=r*segments+(n+1)%segments,c=(r+1)*segments+(n+1)%segments,d=(r+1)*segments+n;
  faces.push([a,b,c],[a,c,d]);
 }
 const cap=(ring:number,reverse:boolean)=>{
  const center=rings[ring];
  const index=vertices.length;
  vertices.push(movePoint([0,0,center.z],x,z,angle));
  for(let n=0;n<segments;n++){
   const a=ring*segments+n,b=ring*segments+(n+1)%segments;
   faces.push(reverse?[index,b,a]:[index,a,b]);
  }
 };
 cap(0,true);cap(rings.length-1,false);
 return {...mesh(vertices,faces,color,name,opacity),lighting:METAL};
}

function finMesh(x:number,z:number,angle:number):Row {
 const vertices:Point3[]=[],faces:Face[]=[];
 // Four swept fins sit behind the main hull. Their dark faces contrast with
 // the bright edge caps and help the rocket read clearly as it rotates.
 for(let n=0;n<4;n++){
  const theta=n*Math.PI/2,c=Math.cos(theta),s=Math.sin(theta);
  const profile:Array<[number,number]>=[[.25,-.14],[.53,-.24],[.51,-.39],[.29,-.33]];
  const start=vertices.length;
  for(const side of [-.035,.035]) for(const [r,height] of profile){
   const tangentX=-s*side,tangentY=c*side;
   vertices.push(movePoint([r*c+tangentX,r*s+tangentY,height],x,z,angle));
  }
  // Front and rear panels, followed by the thin swept edges.
  faces.push([start,start+1,start+2],[start,start+2,start+3]);
  faces.push([start+4,start+6,start+5],[start+4,start+7,start+6]);
  for(let edge=0;edge<4;edge++){
   const next=(edge+1)%4,a=start+edge,b=start+next;
   faces.push([a,b,b+4],[a,b+4,a+4]);
  }
 }
 return mesh(vertices,faces,'#74777a','Stabilizer fins');
}

function landingGear(x:number,z:number,angle:number):Row[] {
 const struts:LinePoint[]=[],feet:Point3[]=[],braces:LinePoint[]=[];
 for(let n=0;n<4;n++){
  const theta=n*Math.PI/2,c=Math.cos(theta),s=Math.sin(theta);
  const radial=(radius:number,height:number):Point3=>movePoint([radius*c,radius*s,height],x,z,angle);
  const shoulder=radial(.24,-.18),knee=radial(.39,-.31),toe=radial(.56,-.51);
  const braceTop=radial(.27,-.26),braceEnd=radial(.52,-.44);
  struts.push(shoulder,knee,toe,null);
  braces.push(braceTop,braceEnd,null);
  feet.push(radial(.57,-.52));
 }
 const line=(points:LinePoint[],color:string,width:number,name:string):Row=>({
  type:'scatter3d',x:points.map(p=>p?.[0]??null),y:points.map(p=>p?.[1]??null),z:points.map(p=>p?.[2]??null),
  mode:'lines',line:{color,width},name,showlegend:false,
 });
 const footPoints=feet.map(p=>p[0]);
 return [
  line(struts,'#d0d2d3',5,'Landing struts'),
  line(braces,'#85898c',3,'Gear braces'),
  {type:'scatter3d',x:footPoints,y:feet.map(p=>p[1]),z:feet.map(p=>p[2]),mode:'markers',
   marker:{size:3,color:'#eeeeee',line:{color:'#55595c',width:1}},name:'Landing feet',showlegend:false},
 ];
}

function surfaceDetails(x:number,z:number,angle:number):Row[] {
 const windowCenter=movePoint([0,-.318,.155],x,z,angle);
 const portholes=[-.12,.01,.13].map(height=>movePoint([.12,-.296,height],x,z,angle));
 const stripe:[number,number,number]=movePoint([.025,-.322,.155],x,z,angle);
 return [
  {type:'scatter3d',x:[windowCenter[0]],y:[windowCenter[1]],z:[windowCenter[2]],mode:'markers',
   marker:{size:6,color:'#f0f0f0',line:{color:'#45494c',width:1}},name:'Cockpit rim',showlegend:false},
  {type:'scatter3d',x:[windowCenter[0]],y:[windowCenter[1]-.004],z:[windowCenter[2]],mode:'markers',
   marker:{size:3,color:'#25282b',line:{color:'#b7b9bb',width:1}},name:'Cockpit glass',showlegend:false},
  {type:'scatter3d',x:portholes.map(p=>p[0]),y:portholes.map(p=>p[1]),z:portholes.map(p=>p[2]),mode:'markers',
   marker:{size:2,color:'#45494c',line:{color:'#d1d3d4',width:1}},name:'Hull instruments',showlegend:false},
  {type:'scatter3d',x:[stripe[0]],y:[stripe[1]-.004],z:[stripe[2]],mode:'markers',
   marker:{size:2,color:'#ffffff'},name:'Hull insignia',showlegend:false},
 ];
}

function coneTrace(start:Point3,end:Point3,radiusA:number,radiusB:number,color:string,name:string,x:number,z:number,angle:number,opacity=1):Row {
 const dx=end[0]-start[0],dy=end[1]-start[1],dz=end[2]-start[2];
 const length=Math.hypot(dx,dy,dz)||1,axis:Point3=[dx/length,dy/length,dz/length];
 const reference:Point3=Math.abs(axis[2])<.9?[0,0,1]:[0,1,0];
 const cross=(a:Point3,b:Point3):Point3=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
 const normalise=(v:Point3):Point3=>{const m=Math.hypot(...v)||1;return [v[0]/m,v[1]/m,v[2]/m]};
 const u=normalise(cross(axis,reference)),v=normalise(cross(axis,u));
 const vertices:Point3[]=[];
 for(const [center,radius] of [[start,radiusA],[end,radiusB]] as Array<[Point3,number]>){
  for(let n=0;n<SEGMENTS;n++){
   const theta=2*Math.PI*n/SEGMENTS;
   vertices.push(movePoint([center[0]+radius*(Math.cos(theta)*u[0]+Math.sin(theta)*v[0]),center[1]+radius*(Math.cos(theta)*u[1]+Math.sin(theta)*v[1]),center[2]+radius*(Math.cos(theta)*u[2]+Math.sin(theta)*v[2])],x,z,angle,false));
  }
 }
 const faces:Face[]=[];
 for(let n=0;n<SEGMENTS;n++){
  const next=(n+1)%SEGMENTS;
  faces.push([n,next,SEGMENTS+next],[n,SEGMENTS+next,SEGMENTS+n]);
 }
 return {...mesh(vertices,faces,color,name,opacity),lighting:{ambient:.8,diffuse:.4,specular:.2,roughness:.6}};
}

function rocket(x:number,z:number,angle:number,action:number):Row[] {
 const body=ringsTrace([
  {z:-.36,radius:.23},{z:-.31,radius:.29},{z:-.22,radius:.315},
  {z:.17,radius:.315},{z:.25,radius:.29},{z:.29,radius:.255},
 ],'#c6c9cb','Aluminum fuselage',x,z,angle);
 const nose=ringsTrace([
  {z:.24,radius:.275},{z:.31,radius:.26},{z:.43,radius:.21},
  {z:.56,radius:.13},{z:.65,radius:.045},{z:.67,radius:.008},
 ],'#f0f1f2','Nose cone',x,z,angle);
 const nozzle=ringsTrace([
  {z:-.29,radius:.19},{z:-.36,radius:.21},{z:-.46,radius:.145},
  {z:-.50,radius:.12},
 ],'#55595c','Main engine bell',x,z,angle);
 const band=ringsTrace([{z:-.06,radius:.321},{z:.015,radius:.321}], '#35383b','Black mission band',x,z,angle);
 // Thin silver rails frame the dark service band.
 const rails=ringsTrace([{z:-.088,radius:.324},{z:-.077,radius:.324}], '#e8e9ea','Lower hull rail',x,z,angle);
 const upperRail=ringsTrace([{z:.027,radius:.324},{z:.04,radius:.324}], '#e8e9ea','Upper hull rail',x,z,angle);
 const direction=action===1?-1:1;
 const plumeStart:Point3=action===2?[0,0,-.47]:[direction*.31,0,-.08];
 const plumeEnd:Point3=action===2?[0,0,-.87]:[direction*.65,0,-.08];
 const active=action===2||action===1||action===3;
 const flame=active?coneTrace(plumeStart,plumeEnd,.065,.16,'#eeeeee','Engine plume',x,z,angle,.62):coneTrace([0,0,-.47],[0,0,-.48],.001,.001,'#eeeeee','Engine plume',x,z,angle,0);
 const core=active?coneTrace(plumeStart,action===2?[0,0,-.76]:[direction*.54,0,-.08],.038,.07,'#ffffff','Exhaust core',x,z,angle,.95):coneTrace([0,0,-.47],[0,0,-.48],.001,.001,'#ffffff','Exhaust core',x,z,angle,0);
 return [
  body,nose,nozzle,band,rails,upperRail,finMesh(x,z,angle),
  ...landingGear(x,z,angle),...surfaceDetails(x,z,angle),flame,core,
 ];
}

export function savedFlightReplay(saved:SavedFlight):Replay {
 if(saved.schema!==1||!saved.rows.length||saved.steps!==saved.rows.length-1)throw Error('This saved flight has an unsupported format.');
 const flight:Row[]=saved.rows.map(row=>Object.fromEntries(saved.columns.map((key,i)=>[key,row[i]])));
 const x=flight.map(r=>r.x*saved.world_scale[0]),z=flight.map(r=>r.height*saved.world_scale[1]+saved.world_scale[2]);
 const count=Math.min(120,flight.length),positions=Array.from({length:count},(_,i)=>Math.round(i*(flight.length-1)/Math.max(1,count-1)));
 const at=(n:number)=>rocket(x[n],z[n],flight[n].angle,flight[n].action);
 const model=at(0),frames=positions.map(n=>({name:String(n),data:at(n),traces:model.map((_,i)=>i+1)}));
 const duration=Math.max(40,Math.round(1000*saved.steps/50/Math.max(1,count-1)));
 const axis={gridcolor:'#333333',zerolinecolor:'#555555',color:'#999999',showbackground:false};
 const figure={data:[{type:'mesh3d',x:[saved.pad[0],saved.pad[1],saved.pad[1],saved.pad[0]],y:[-.8,-.8,.8,.8],z:[0,0,0,0],i:[0,0],j:[1,2],k:[2,3],color:'#a5a5a5',opacity:.9,name:'Landing pad',showlegend:false},...model,{type:'scatter3d',x,y:x.map(()=>0),z,mode:'lines',line:{color:'#999999',width:3},name:'Flight path',showlegend:false}],frames,layout:{template:null,paper_bgcolor:'#111111',plot_bgcolor:'#111111',font:{color:'#d4d4d4'},showlegend:false,margin:{l:20,r:20,t:15,b:60},scene:{bgcolor:'#111111',aspectmode:'manual',aspectratio:{x:1.5,y:.55,z:1.7},camera:{eye:{x:1.1,y:-1.6,z:.8}},xaxis:{...axis,title:{text:'Horizontal position'},range:[Math.min(-10,...x.map(v=>v-1)),Math.max(10,...x.map(v=>v+1))]},yaxis:{...axis,title:{text:'Depth (illustration)'},range:[-2,2]},zaxis:{...axis,title:{text:'Height above pad'},range:[Math.min(-1,...z.map(v=>v-1)),Math.max(11,...z.map(v=>v+2.25))]},lightposition:{x:-100,y:-200,z:300}},updatemenus:[{type:'buttons',showactive:false,x:0,y:0,bgcolor:'#222222',bordercolor:'#444444',font:{color:'#dddddd'},buttons:[{label:'▶ Play',method:'animate',args:[null,{frame:{duration,redraw:true},transition:{duration:0},fromcurrent:true}]},{label:'❚❚ Pause',method:'animate',args:[[null],{frame:{duration:0,redraw:false},mode:'immediate'}]}]}],sliders:[{active:0,x:.16,len:.8,y:0,bgcolor:'#555555',activebgcolor:'#eeeeee',bordercolor:'#444444',font:{color:'#999999'},currentvalue:{prefix:'Step '},steps:positions.map(n=>({label:String(n),method:'animate',args:[[String(n)],{frame:{duration:0,redraw:true},mode:'immediate'}]}))}]}};
 return {figure,flight,outcome:saved.outcome,native_return:saved.native_return,shaped_return:saved.shaped_return,steps:saved.steps};
}
