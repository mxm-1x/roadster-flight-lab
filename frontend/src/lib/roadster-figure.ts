import type {Row} from './api';
import physicsConfigV4 from '../../../rocket_landing/configs/roadster_v4.json';
import physicsConfigV5 from '../../../rocket_landing/configs/roadster_v5.json';
import physicsConfigV6 from '../../../rocket_landing/configs/roadster_v6.json';
import physicsConfigV7 from '../../../rocket_landing/configs/roadster_v7.json';
import {buildRampGeometry} from './ramp-geometry';

export type SavedRoadster = {
  schema:number;simulator_version:number;columns:string[];rows:number[][];phases:string[];
  ramp_radius_m?:number;ramp_angle_rad?:number;ramp_start_x?:number;ramp_end_x?:number;ramp_top_y?:number;
  ramp_vertical_stem_length_m?:number;ramp_top_curve_radius_m?:number;ramp_top_curve_angle_deg?:number;
  landing_zone_x:[number,number];landing_zone_z?:[number,number];outcome:string;
  native_return:number;shaped_return:number;steps:number;
  metrics:Record<string,number|number[]|null>;source:Record<string,unknown>;
};
type Point=[number,number,number];
type Face=[number,number,number];
const SIDES=12;
const LIGHTING={ambient:.5,diffuse:.8,specular:.5,roughness:.35,fresnel:.12};

function transform(point:Point,x:number,z:number,heading:number):Point{
  const c=Math.cos(heading),s=Math.sin(heading);
  return [x+point[0]*c-point[2]*s,point[1],z+point[0]*s+point[2]*c];
}
function mesh(points:Point[],faces:Face[],color:string,name:string,opacity=1):Row{
  return {type:'mesh3d',x:points.map(p=>p[0]),y:points.map(p=>p[1]),z:points.map(p=>p[2]),
    i:faces.map(f=>f[0]),j:faces.map(f=>f[1]),k:faces.map(f=>f[2]),color,opacity,name,
    flatshading:false,lighting:LIGHTING,showlegend:false};
}
function extrudedPolygon(profile:Array<[number,number]>,x:number,z:number,heading:number,width:number,color:string,name:string):Row{
  const points:Point[]=[];
  for(const side of [-width,width]) for(const [px,pz] of profile) points.push(transform([px,side,pz],x,z,heading));
  const faces:Face[]=[];const n=profile.length;
  for(let i=1;i<n-1;i++){faces.push([0,i,i+1],[n,n+i+1,n+i]);}
  for(let i=0;i<n;i++){const j=(i+1)%n;faces.push([i,j,n+j],[i,n+j,n+i]);}
  return mesh(points,faces,color,name);
}
function wheel(x:number,z:number,heading:number,cx:number,cy:number):Row{
  const points:Point[]=[];
  for(const side of [-.11,.11]) for(let i=0;i<SIDES;i++){
    const a=2*Math.PI*i/SIDES;
    points.push(transform([cx+.29*Math.cos(a),cy+side,-.2+.29*Math.sin(a)],x,z,heading));
  }
  const faces:Face[]=[];
  for(let i=0;i<SIDES;i++){const j=(i+1)%SIDES;faces.push([i,j,SIDES+j],[i,SIDES+j,SIDES+i]);}
  for(let i=1;i<SIDES-1;i++){faces.push([0,i,i+1],[SIDES,SIDES+i+1,SIDES+i]);}
  return mesh(points,faces,'#171717','Road wheel');
}
function lineTrace(points:Point[],color:string,width:number,name:string,visible=true):Row{
  return {type:'scatter3d',x:points.map(p=>p?.[0]??null),y:points.map(p=>p?.[1]??null),z:points.map(p=>p?.[2]??null),
    mode:'lines',line:{color,width},name,visible,showlegend:false};
}
function car(x:number,z:number,heading:number,hjet:number,vjet:number,tjet:number):Row[]{
  const body=extrudedPolygon([[-1.72,-.17],[-1.58,.02],[-1.05,.16],[-.58,.52],[.26,.52],[.72,.25],[1.47,.17],[1.72,.03],[1.62,-.18]],x,z,heading,.53,'#dedede','Electric sports car');
  const cabin=extrudedPolygon([[-.66,.22],[-.34,.46],[.18,.46],[.55,.25]],x,z,heading,.445,'#202020','Dark glass canopy');
  const trim=extrudedPolygon([[-1.48,-.1],[-1.3,-.07],[1.48,-.06],[1.57,-.12]],x,z,heading,.54,'#f4f4f4','Lower body trim');
  const wheels=[wheel(x,z,heading,-1.0,-.13),wheel(x,z,heading,-1.0,.13),wheel(x,z,heading,1.05,-.13),wheel(x,z,heading,1.05,.13)];
  const h0=transform([0,0,-.08],x,z,heading),h1=transform([-Math.sign(hjet||1)*(.45+Math.abs(hjet)*.7),0,-.08],x,z,heading);
  const v0=transform([-.35,0,-.17],x,z,heading),v1=transform([-.35,0,-.17-Math.max(0,vjet)*.95],x,z,heading);
  const t0=transform([Math.sign(tjet||1)*.95,0,.1],x,z,heading),t1=transform([Math.sign(tjet||1)*(.95+Math.abs(tjet)*.55),0,.1],x,z,heading);
  const hPlume=lineTrace([h0,h1],'#c8c8c8',7,'Horizontal jet plume',Math.abs(hjet)>.03);
  const vPlume=lineTrace([v0,v1],'#ffffff',8,'Vertical jet plume',vjet>.03);
  const tPlume=lineTrace([t0,t1],'#a9a9a9',5,'Attitude jet plume',Math.abs(tjet)>.03);
  return [body,cabin,trim,...wheels,hPlume,vPlume,tPlume];
}

function roadsterFlightFigureV3(saved:SavedRoadster):{data:Row[];frames:Row[];layout:Row;flight:Row[]} {
  if(saved.schema!==1||saved.simulator_version!==3||!saved.rows.length||saved.steps!==saved.rows.length-1)throw Error('Unsupported saved Roadster trajectory.');
  const flight:Row[]=saved.rows.map(row=>Object.fromEntries(saved.columns.map((key,i)=>[key,row[i]])));
  const x=flight.map(row=>Number(row.x)),z=flight.map(row=>Number(row.z));
  const cx=saved.landing_zone_x[0],padEnd=saved.landing_zone_x[1],radius=saved.ramp_radius_m??10;
  const rampX:number[]=[],rampY:number[]=[],rampZ:number[]=[];
  for(let i=0;i<=72;i++){
    const q=Math.PI*i/72;
    rampX.push(radius*Math.sin(q));rampY.push(0);rampZ.push(radius*(1-Math.cos(q))+.65);
  }
  const pathCount=Math.min(160,flight.length);
  const frameSteps=Array.from({length:pathCount},(_,i)=>Math.round(i*(flight.length-1)/Math.max(1,pathCount-1)));
  const at=(n:number)=>car(x[n],z[n],Number(flight[n].heading),Number(flight[n].horizontal_jet),Number(flight[n].vertical_jet),Number(flight[n].attitude_jet));
  const dynamic=at(0);
  const fixed:Row[]=[
    {type:'scatter3d',x:[-32,24],y:[0,0],z:[0,0],mode:'lines',line:{color:'#555555',width:5},name:'Ground plane',showlegend:false},
    {type:'scatter3d',x:rampX,y:rampY,z:rampZ,mode:'lines',line:{color:'#c8c8c8',width:13},name:'Scripted ramp fixture',showlegend:false},
    {type:'scatter3d',x:[cx,padEnd],y:[0,0],z:[.03,.03],mode:'lines',line:{color:'#ffffff',width:13},name:'Landing zone',showlegend:false},
    {type:'scatter3d',x,y:x.map(()=>0),z,mode:'lines',line:{color:'#8a8a8a',width:4},name:'Flight path',showlegend:false},
  ];
  const start= fixed.length;
  const frames=frameSteps.map(n=>({name:String(n),data:at(n),traces:dynamic.map((_,i)=>start+i)}));
  const allX=[...x,-2,padEnd,radius+2],allZ=[...z,-1,radius*2+4];
  const axis={gridcolor:'#303030',zerolinecolor:'#555555',color:'#aaaaaa',showbackground:false};
  const layout:Row={template:null,paper_bgcolor:'#101010',plot_bgcolor:'#101010',font:{color:'#dedede'},
    showlegend:false,margin:{l:10,r:10,t:12,b:55},
    scene:{bgcolor:'#101010',aspectmode:'manual',aspectratio:{x:1.45,y:.32,z:1.1},camera:{eye:{x:1.15,y:-1.75,z:.9}},
      xaxis:{...axis,title:{text:'Horizontal distance (m)'},range:[Math.min(-28,...allX),Math.max(14,...allX)]},
      yaxis:{...axis,title:{text:'Depth (illustrative)'},range:[-4,4]},
      zaxis:{...axis,title:{text:'Height (m)'},range:[-1,Math.max(24,...allZ)]},lightposition:{x:-100,y:-200,z:300}},
    updatemenus:[{type:'buttons',showactive:false,x:0,y:0,bgcolor:'#222222',bordercolor:'#444444',font:{color:'#dddddd'},buttons:[
      {label:'▶ Play',method:'animate',args:[null,{frame:{duration:45,redraw:true},transition:{duration:0},fromcurrent:true}]},
      {label:'❚❚ Pause',method:'animate',args:[[null],{frame:{duration:0,redraw:false},mode:'immediate'}]},
    ]}],
    sliders:[{active:0,x:.16,len:.8,y:0,bgcolor:'#555555',activebgcolor:'#eeeeee',bordercolor:'#444444',font:{color:'#999999'},currentvalue:{prefix:'Step '},
      steps:frameSteps.map(n=>({label:String(n),method:'animate',args:[[String(n)],{frame:{duration:0,redraw:true},mode:'immediate'}]}))}],
  };
  return {data:[...fixed,...dynamic],frames,layout,flight};
}

type Pose3D={x:number;y:number;z:number;rotation_qx:number;rotation_qy:number;rotation_qz:number;rotation_qw:number;
  ramp_drive:number;forward_thrust:number;vertical_thrust:number;lateral_thrust:number;pitch_thrusters:number;roll_thrusters:number;yaw_thrusters:number};
function rotate3(point:Point,q:[number,number,number,number]):Point{
  const [x,y,z,w]=q,tx=2*(y*point[2]-z*point[1]),ty=2*(z*point[0]-x*point[2]),tz=2*(x*point[1]-y*point[0]);
  return [point[0]+w*tx+(y*tz-z*ty),point[1]+w*ty+(z*tx-x*tz),point[2]+w*tz+(x*ty-y*tx)];
}
function point3(local:Point,pose:Pose3D):Point{
  const rotated=rotate3(local,[pose.rotation_qx,pose.rotation_qy,pose.rotation_qz,pose.rotation_qw]);
  // Rapier uses Y-up; the viewer keeps Plotly's conventional Z-up scene.
  return [pose.x+rotated[0],pose.z+rotated[2],pose.y+rotated[1]];
}
function extruded3D(profile:Array<[number,number]>,pose:Pose3D,width:number,color:string,name:string):Row{
  const points:Point[]=[];
  for(const side of [-width,width])for(const [x,y] of profile)points.push(point3([x,y,side],pose));
  const faces:Face[]=[],n=profile.length;
  for(let i=1;i<n-1;i++)faces.push([0,i,i+1],[n,n+i+1,n+i]);
  for(let i=0;i<n;i++){const j=(i+1)%n;faces.push([i,j,n+j],[i,n+j,n+i]);}
  return mesh(points,faces,color,name);
}
function wheel3D(pose:Pose3D,cx:number,cy:number,cz:number,radius=.29,halfWidth=.12):Row{
  const points:Point[]=[];const sides=[cz-halfWidth,cz+halfWidth];
  for(const side of sides)for(let i=0;i<SIDES;i++){const a=2*Math.PI*i/SIDES;points.push(point3([cx+radius*Math.cos(a),cy+radius*Math.sin(a),side],pose));}
  const faces:Face[]=[];for(let i=0;i<SIDES;i++){const j=(i+1)%SIDES;faces.push([i,j,SIDES+j],[i,SIDES+j,SIDES+i]);}
  for(let i=1;i<SIDES-1;i++)faces.push([0,i,i+1],[SIDES,SIDES+i+1,SIDES+i]);
  return mesh(points,faces,'#171717','Wheel contact proxy');
}
function car3D(pose:Pose3D):Row[]{
  const body=extruded3D([[-1.72,-.17],[-1.58,.02],[-1.05,.16],[-.58,.52],[.26,.52],[.72,.25],[1.47,.17],[1.72,.03],[1.62,-.18]],pose,.53,'#dedede','Roadster rigid body');
  const cabin=extruded3D([[-.66,.22],[-.34,.46],[.18,.46],[.55,.25]],pose,.445,'#202020','Dark glass canopy');
  const trim=extruded3D([[-1.48,-.1],[-1.3,-.07],[1.48,-.06],[1.57,-.12]],pose,.54,'#f4f4f4','Lower body trim');
  const wheels=[wheel3D(pose,-1,-.13,-.78),wheel3D(pose,-1,-.13,.78),wheel3D(pose,1.05,-.13,-.78),wheel3D(pose,1.05,-.13,.78)];
  const lift=(pose.vertical_thrust+1)/2;
  const thrust=(local:Point,axis:Point,amount:number,name:string)=>{
    const start=point3(local,pose),end=point3([local[0]+axis[0]*amount,local[1]+axis[1]*amount,local[2]+axis[2]*amount],pose);
    return lineTrace([start,end],'#eeeeee',7,name,amount>.035);
  };
  const rampDrive=(pose.ramp_drive+1)/2;
  const plumes=[
    thrust([-1.4,-.12,0],[-1,0,0],rampDrive*.75,'Launch drive plume'),
    thrust([-.4,-.15,0],[-Math.sign(pose.forward_thrust||1),0,0],Math.abs(pose.forward_thrust)*.09,'Forward thruster plume'),
    thrust([0,-.25,0],[0,-1,0],lift*.9,'Vertical thruster plume'),
    thrust([0,-.15,-.3],[0,0,-Math.sign(pose.lateral_thrust||1)],Math.abs(pose.lateral_thrust)*.1,'Lateral thruster plume'),
  ];
  return [body,cabin,trim,...wheels,...plumes];
}
function staticBox3D(center:Point,half:Point,angle:number,color:string,name:string):Row{
  const q:[number,number,number,number]=[0,0,Math.sin(angle/2),Math.cos(angle/2)];
  const points:Point[]=[];for(const x of [-half[0],half[0]])for(const y of [-half[1],half[1]])for(const z of [-half[2],half[2]]){
    const p=rotate3([x,y,z],q);points.push([center[0]+p[0],center[2]+p[2],center[1]+p[1]]);
  }
  const faces:Face[]=[[0,1,3],[0,3,2],[4,6,7],[4,7,5],[0,4,5],[0,5,1],[2,3,7],[2,7,6],[0,2,6],[0,6,4],[1,5,7],[1,7,3]];
  return mesh(points,faces,color,name);
}
function roadsterFlightFigureV4(saved:SavedRoadster):{data:Row[];frames:Row[];layout:Row;flight:Row[]}{
  if(saved.schema!==2||saved.simulator_version!==4||!saved.rows.length||saved.steps!==saved.rows.length-1)throw Error('Unsupported v4 Rapier trajectory.');
  const flight:Row[]=saved.rows.map(row=>Object.fromEntries(saved.columns.map((key,i)=>[key,row[i]])));
  const pose=(row:Row):Pose3D=>({x:Number(row.x),y:Number(row.y),z:Number(row.z),rotation_qx:Number(row.rotation_qx),rotation_qy:Number(row.rotation_qy),
    rotation_qz:Number(row.rotation_qz),rotation_qw:Number(row.rotation_qw),forward_thrust:Number(row.forward_thrust),vertical_thrust:Number(row.vertical_thrust),
    ramp_drive:Number(row.ramp_drive),lateral_thrust:Number(row.lateral_thrust),pitch_thrusters:Number(row.pitch_thrusters),roll_thrusters:Number(row.roll_thrusters),yaw_thrusters:Number(row.yaw_thrusters)});
  const x=flight.map(row=>Number(row.x)),y=flight.map(row=>Number(row.y)),z=flight.map(row=>Number(row.z));
  const centerX=(saved.landing_zone_x[0]+saved.landing_zone_x[1])/2,zoneZ=saved.landing_zone_z??[-3.5,3.5],centerZ=(zoneZ[0]+zoneZ[1])/2;
  const rampAngle=saved.ramp_angle_rad??0.384,rampStart=saved.ramp_start_x??-2,rampEnd=saved.ramp_end_x??15,rampTop=saved.ramp_top_y??7.5;
  const rampCenterX=(rampStart+rampEnd)/2;
  const rampCenterY=rampTop/2;
  const fixed:Row[]=[
    staticBox3D([rampCenterX,rampCenterY,0],[physicsConfigV4.ramp_length_m/2,physicsConfigV4.ramp_thickness_m/2,
      physicsConfigV4.ramp_half_width_m],rampAngle,'#777777','Rigid-body launch ramp'),
    {type:'scatter3d',x:[-55,75],y:[0,0],z:[0,0],mode:'lines',line:{color:'#555555',width:5},name:'Ground collision surface',showlegend:false},
    {type:'scatter3d',x:[saved.landing_zone_x[0],saved.landing_zone_x[1],saved.landing_zone_x[1],saved.landing_zone_x[0],saved.landing_zone_x[0]],
      y:[zoneZ[0],zoneZ[0],zoneZ[1],zoneZ[1],zoneZ[0]],z:[.02,.02,.02,.02,.02],mode:'lines',line:{color:'#ffffff',width:8},name:'Safe landing region',showlegend:false},
    {type:'scatter3d',x,y:z,z:y,mode:'lines',line:{color:'#a0a0a0',width:4},name:'Rapier rigid-body path',showlegend:false},
  ];
  const frameSteps=Array.from({length:Math.min(160,flight.length)},(_,i)=>Math.round(i*(flight.length-1)/Math.max(1,Math.min(160,flight.length)-1)));
  const start=fixed.length,dynamic=car3D(pose(flight[0]));
  const frames=frameSteps.map(n=>({name:String(n),data:car3D(pose(flight[n])),traces:dynamic.map((_,i)=>start+i)}));
  const allX=[...x,saved.landing_zone_x[0],saved.landing_zone_x[1],rampStart,rampEnd],allY=[...y,0,rampTop];
  const axis={gridcolor:'#303030',zerolinecolor:'#555555',color:'#aaaaaa',showbackground:false};
  const layout:Row={template:null,paper_bgcolor:'#101010',plot_bgcolor:'#101010',font:{color:'#dedede'},showlegend:false,margin:{l:10,r:10,t:12,b:55},
    scene:{bgcolor:'#101010',aspectmode:'manual',aspectratio:{x:1.5,y:.6,z:.9},camera:{eye:{x:1.35,y:-1.8,z:1.15},up:{x:0,y:0,z:1}},
      xaxis:{...axis,title:{text:'Downrange (m)'},range:[Math.min(-20,...allX)-3,Math.max(40,...allX)+4]},
      yaxis:{...axis,title:{text:'Lateral (m)'},range:[Math.min(-10,...z)-2,Math.max(10,...z)+2]},
      zaxis:{...axis,title:{text:'Height (m)'},range:[-1,Math.max(20,...allY)+3]},lightposition:{x:-100,y:-200,z:300}},
    updatemenus:[{type:'buttons',showactive:false,x:0,y:0,bgcolor:'#222222',bordercolor:'#444444',font:{color:'#dddddd'},buttons:[
      {label:'▶ Play',method:'animate',args:[null,{frame:{duration:45,redraw:true},transition:{duration:0},fromcurrent:true}]},
      {label:'❚❚ Pause',method:'animate',args:[[null],{frame:{duration:0,redraw:false},mode:'immediate'}]},
    ]}],
    sliders:[{active:0,x:.16,len:.8,y:0,bgcolor:'#555555',activebgcolor:'#eeeeee',bordercolor:'#444444',font:{color:'#999999'},currentvalue:{prefix:'Step '},
      steps:frameSteps.map(n=>({label:String(n),method:'animate',args:[[String(n)],{frame:{duration:0,redraw:true},mode:'immediate'}]}))}],
  };
  return {data:[...fixed,...dynamic],frames,layout,flight};
}

function roadsterFlightFigureV5(saved:SavedRoadster):{data:Row[];frames:Row[];layout:Row;flight:Row[]} {
  if(saved.schema!==3||saved.simulator_version!==5||!saved.rows.length||saved.steps!==saved.rows.length-1)
    throw Error('Unsupported v5 Rapier half-circle trajectory.');
  const flight:Row[]=saved.rows.map(row=>Object.fromEntries(saved.columns.map((key,i)=>[key,row[i]])));
  const pose=(row:Row):Pose3D=>({x:Number(row.x),y:Number(row.y),z:Number(row.z),
    rotation_qx:Number(row.rotation_qx),rotation_qy:Number(row.rotation_qy),rotation_qz:Number(row.rotation_qz),rotation_qw:Number(row.rotation_qw),
    ramp_drive:Number(row.ramp_drive),forward_thrust:Number(row.forward_thrust),vertical_thrust:Number(row.vertical_thrust),
    lateral_thrust:Number(row.lateral_thrust),pitch_thrusters:Number(row.pitch_thrusters),roll_thrusters:Number(row.roll_thrusters),yaw_thrusters:Number(row.yaw_thrusters)});
  const x=flight.map(row=>Number(row.x)),y=flight.map(row=>Number(row.y)),z=flight.map(row=>Number(row.z));
  const radius=saved.ramp_radius_m??physicsConfigV5.ramp_radius_m,startX=saved.ramp_start_x??physicsConfigV5.ramp_start_x_m;
  const clearance=physicsConfigV5.ramp_vehicle_clearance_m,halfWidth=physicsConfigV5.ramp_half_width_m;
  const thickness=physicsConfigV5.ramp_thickness_m,segments=physicsConfigV5.ramp_segments;
  const surfaceRadius=radius+clearance,centerY=surfaceRadius,outerRadius=surfaceRadius+thickness;
  const vertices:Point[]=[],faces:Face[]=[];
  for(let i=0;i<=segments;i++){
    const theta=Math.PI*i/segments,s=Math.sin(theta),c=Math.cos(theta);
    const ix=startX+surfaceRadius*s,iy=centerY-surfaceRadius*c,ox=startX+outerRadius*s,oy=centerY-outerRadius*c;
    vertices.push([ix,-halfWidth,iy],[ix,halfWidth,iy],[ox,-halfWidth,oy],[ox,halfWidth,oy]);
  }
  for(let i=0;i<segments;i++){
    const a=4*i,b=a+4;
    faces.push([a,a+1,b+1],[a,b+1,b]);
    faces.push([a+2,b+2,b+3],[a+2,b+3,a+3]);
    faces.push([a+1,b+3,b+1],[a+1,a+3,b+3]);
    faces.push([a,b,b+2],[a,b+2,a+2]);
  }
  const ramp=mesh(vertices,faces,'#777777','Tesla half-circle collision ramp');
  const zoneZ=saved.landing_zone_z??[-3.5,3.5];
  const fixed:Row[]=[ramp,
    {type:'scatter3d',x:[-60,80],y:[0,0],z:[0,0],mode:'lines',line:{color:'#555555',width:5},name:'Ground collision surface',showlegend:false},
    {type:'scatter3d',x:[saved.landing_zone_x[0],saved.landing_zone_x[1],saved.landing_zone_x[1],saved.landing_zone_x[0],saved.landing_zone_x[0]],
      y:[zoneZ[0],zoneZ[0],zoneZ[1],zoneZ[1],zoneZ[0]],z:[.02,.02,.02,.02,.02],mode:'lines',line:{color:'#ffffff',width:8},name:'Safe landing region',showlegend:false},
    {type:'scatter3d',x,y:z,z:y,mode:'lines',line:{color:'#a0a0a0',width:4},name:'Rapier rigid-body path',showlegend:false},
  ];
  const count=Math.min(180,flight.length),frameSteps=Array.from({length:count},(_,i)=>Math.round(i*(flight.length-1)/Math.max(1,count-1)));
  const start=fixed.length,dynamic=car3D(pose(flight[0]));
  const frames=frameSteps.map(n=>({name:String(n),data:car3D(pose(flight[n])),traces:dynamic.map((_,i)=>start+i)}));
  const allX=[...x,saved.landing_zone_x[0],saved.landing_zone_x[1],startX,startX+outerRadius];
  const allY=[...y,0,2*surfaceRadius+thickness];
  const axis={gridcolor:'#303030',zerolinecolor:'#555555',color:'#aaaaaa',showbackground:false};
  const layout:Row={template:null,paper_bgcolor:'#101010',plot_bgcolor:'#101010',font:{color:'#dedede'},showlegend:false,margin:{l:10,r:10,t:12,b:55},
    scene:{bgcolor:'#101010',aspectmode:'manual',aspectratio:{x:.78,y:.55,z:1.35},camera:{eye:{x:1.55,y:-1.8,z:1.1},up:{x:0,y:0,z:1}},
      xaxis:{...axis,title:{text:'Downrange (m)'},range:[Math.min(-38,...allX)-4,Math.max(18,...allX)+5]},
      yaxis:{...axis,title:{text:'Lateral (m)'},range:[-10,10]},
      zaxis:{...axis,title:{text:'Height (m)'},range:[-1,Math.max(25,...allY)+3]},lightposition:{x:-100,y:-200,z:300}},
    updatemenus:[{type:'buttons',showactive:false,x:0,y:0,bgcolor:'#222222',bordercolor:'#444444',font:{color:'#dddddd'},buttons:[
      {label:'▶ Play',method:'animate',args:[null,{frame:{duration:45,redraw:true},transition:{duration:0},fromcurrent:true}]},
      {label:'❚❚ Pause',method:'animate',args:[[null],{frame:{duration:0,redraw:false},mode:'immediate'}]},
    ]}],
    sliders:[{active:0,x:.16,len:.8,y:0,bgcolor:'#555555',activebgcolor:'#eeeeee',bordercolor:'#444444',font:{color:'#999999'},currentvalue:{prefix:'Step '},
      steps:frameSteps.map(n=>({label:String(n),method:'animate',args:[[String(n)],{frame:{duration:0,redraw:true},mode:'immediate'}]}))}],
  };
  return {data:[...fixed,...dynamic],frames,layout,flight};
}

function roadsterFlightFigureV6(saved:SavedRoadster):{data:Row[];frames:Row[];layout:Row;flight:Row[]} {
  if(saved.schema!==4||![6,7].includes(saved.simulator_version)||!saved.rows.length||saved.steps!==saved.rows.length-1)
    throw Error('Unsupported 3D curved-shoulder ramp trajectory.');
  if(saved.simulator_version===7&&!['wheel_0_x','wheel_3_qw','actual_lift_n'].every(field=>saved.columns.includes(field)))
    throw Error('The v7 recording is missing wheel or thrust telemetry.');
  const flight:Row[]=saved.rows.map(row=>Object.fromEntries(saved.columns.map((key,i)=>[key,row[i]])));
  const pose=(row:Row):Pose3D=>({x:Number(row.x),y:Number(row.y),z:Number(row.z),
    rotation_qx:Number(row.rotation_qx),rotation_qy:Number(row.rotation_qy),rotation_qz:Number(row.rotation_qz),rotation_qw:Number(row.rotation_qw),
    ramp_drive:Number(row.ramp_drive),forward_thrust:Number(row.forward_thrust),vertical_thrust:Number(row.vertical_thrust),
    lateral_thrust:Number(row.lateral_thrust),pitch_thrusters:Number(row.pitch_thrusters),roll_thrusters:Number(row.roll_thrusters),yaw_thrusters:Number(row.yaw_thrusters)});
  const x=flight.map(row=>Number(row.x)),y=flight.map(row=>Number(row.y)),z=flight.map(row=>Number(row.z));
  const config=saved.simulator_version===7?physicsConfigV7:physicsConfigV6;
  const radius=saved.ramp_radius_m??config.ramp_radius_m,startX=saved.ramp_start_x??config.ramp_start_x_m;
  const geometry=buildRampGeometry({startX,radius,clearance:config.ramp_vehicle_clearance_m,
    verticalLength:saved.ramp_vertical_stem_length_m??config.ramp_vertical_stem_length_m,
    topRadius:saved.ramp_top_curve_radius_m??config.ramp_top_curve_radius_m,
    topAngleDeg:saved.ramp_top_curve_angle_deg??config.ramp_top_curve_angle_deg,
    halfWidth:config.ramp_half_width_m,thickness:config.ramp_thickness_m,segments:config.ramp_segments});
  const vertices:Point[]=[];
  for(let i=0;i<geometry.vertices.length;i+=3)vertices.push([geometry.vertices[i],geometry.vertices[i+2],geometry.vertices[i+1]]);
  const faces:Face[]=[];
  for(let i=0;i<geometry.indices.length;i+=3)faces.push([geometry.indices[i],geometry.indices[i+1],geometry.indices[i+2]]);
  const ramp=mesh(vertices,faces,'#777777','Tesla curved-shoulder collision ramp');
  const zoneZ=saved.landing_zone_z??[-3.5,3.5];
  const fixed:Row[]=[ramp,
    {type:'scatter3d',x:[-60,80],y:[0,0],z:[0,0],mode:'lines',line:{color:'#555555',width:5},name:'Ground collision surface',showlegend:false},
    {type:'scatter3d',x:[saved.landing_zone_x[0],saved.landing_zone_x[1],saved.landing_zone_x[1],saved.landing_zone_x[0],saved.landing_zone_x[0]],
      y:[zoneZ[0],zoneZ[0],zoneZ[1],zoneZ[1],zoneZ[0]],z:[.02,.02,.02,.02,.02],mode:'lines',line:{color:'#ffffff',width:8},name:'Safe landing region',showlegend:false},
    {type:'scatter3d',x,y:z,z:y,mode:'lines',line:{color:'#a0a0a0',width:4},name:'Rapier rigid-body path',showlegend:false},
  ];
  const count=Math.min(180,flight.length),frameSteps=Array.from({length:count},(_,i)=>Math.round(i*(flight.length-1)/Math.max(1,count-1)));
  const at=(row:Row)=>{
    const carPose=pose(row);
    if(saved.simulator_version!==7)return car3D(carPose);
    carPose.forward_thrust=Number(row.actual_forward_n)/physicsConfigV7.forward_thrust_max_n;
    carPose.vertical_thrust=2*Number(row.actual_lift_n)/physicsConfigV7.vertical_thrust_max_n-1;
    carPose.lateral_thrust=Number(row.actual_lateral_n)/physicsConfigV7.lateral_thrust_max_n;
    carPose.ramp_drive=-1; // Electric wheel drive has no rocket plume.
    const body=car3D(carPose),wheels=Array.from({length:4},(_,i)=>wheel3D({
      ...carPose,x:Number(row[`wheel_${i}_x`]),y:Number(row[`wheel_${i}_y`]),z:Number(row[`wheel_${i}_z`]),
      rotation_qx:Number(row[`wheel_${i}_qx`]),rotation_qy:Number(row[`wheel_${i}_qy`]),
      rotation_qz:Number(row[`wheel_${i}_qz`]),rotation_qw:Number(row[`wheel_${i}_qw`]),
    },0,0,0,physicsConfigV7.wheel_radius_m,.14));
    return [...body.slice(0,3),...wheels,...body.slice(7)];
  };
  const start=fixed.length,dynamic=at(flight[0]);
  const frames=frameSteps.map(n=>({name:String(n),data:at(flight[n]),traces:dynamic.map((_,i)=>start+i)}));
  const allX=[...x,saved.landing_zone_x[0],saved.landing_zone_x[1],startX,geometry.endX];
  const allY=[...y,0,geometry.topY+config.ramp_thickness_m];
  const axis={gridcolor:'#303030',zerolinecolor:'#555555',color:'#aaaaaa',showbackground:false};
  const layout:Row={template:null,paper_bgcolor:'#101010',plot_bgcolor:'#101010',font:{color:'#dedede'},showlegend:false,margin:{l:10,r:10,t:12,b:55},
    scene:{bgcolor:'#101010',aspectmode:'manual',aspectratio:{x:.78,y:.55,z:1.35},camera:{eye:{x:1.55,y:-1.8,z:1.1},up:{x:0,y:0,z:1}},
      xaxis:{...axis,title:{text:'Downrange (m)'},range:[Math.min(-38,...allX)-4,Math.max(18,...allX)+5]},
      yaxis:{...axis,title:{text:'Lateral (m)'},range:[-10,10]},
      zaxis:{...axis,title:{text:'Height (m)'},range:[-1,Math.max(25,...allY)+3]},lightposition:{x:-100,y:-200,z:300}},
    updatemenus:[{type:'buttons',showactive:false,x:0,y:0,bgcolor:'#222222',bordercolor:'#444444',font:{color:'#dddddd'},buttons:[
      {label:'▶ Play',method:'animate',args:[null,{frame:{duration:45,redraw:true},transition:{duration:0},fromcurrent:true}]},
      {label:'❚❚ Pause',method:'animate',args:[[null],{frame:{duration:0,redraw:false},mode:'immediate'}]},
    ]}],
    sliders:[{active:0,x:.16,len:.8,y:0,bgcolor:'#555555',activebgcolor:'#eeeeee',bordercolor:'#444444',font:{color:'#999999'},currentvalue:{prefix:'Step '},
      steps:frameSteps.map(n=>({label:String(n),method:'animate',args:[[String(n)],{frame:{duration:0,redraw:true},mode:'immediate'}]}))}],
  };
  return {data:[...fixed,...dynamic],frames,layout,flight};
}

export function roadsterFlightFigure(saved:SavedRoadster):{data:Row[];frames:Row[];layout:Row;flight:Row[]} {
  if(saved.simulator_version===6||saved.simulator_version===7)return roadsterFlightFigureV6(saved);
  if(saved.simulator_version===5)return roadsterFlightFigureV5(saved);
  if(saved.simulator_version===4)return roadsterFlightFigureV4(saved);
  if(saved.simulator_version===3)return roadsterFlightFigureV3(saved);
  throw Error(`Unsupported Roadster simulator version ${saved.simulator_version}.`);
}
