import * as RAPIER from '@dimforge/rapier3d-compat';
import physicsConfig from '../../../rocket_landing/configs/roadster_v6.json';
import type {SavedRoadster} from './roadster-figure';
import {buildRampGeometry} from './ramp-geometry';

export type RewardVariant='baseline'|'upright_bonus'|'potential_shaping';
export type Transition={observation:number[];action:number[];oldLogProb:number;reward:number;value:number};
export type BrowserEpisode={episode:number;outcome:string;nativeReturn:number;shapedReturn:number;length:number;safe:boolean;flight:SavedRoadster};
type DenseLayer={input:number;output:number;weights:number[];bias:number[];mw:number[];vw:number[];mb:number[];vb:number[]};
type Network={layers:DenseLayer[];tanhOutput:boolean;step:number};
type Forward={output:number[];activations:number[][]};
type Vec3={x:number;y:number;z:number};
type Quat={x:number;y:number;z:number;w:number};
type Phase='launch'|'flight';
type Scenario={
  world:RAPIER.World;body:RAPIER.RigidBody;target:Vec3;wind:Vec3;launchForce:number;
  rampRadius:number;rampStartX:number;rampEndX:number;rampTopY:number;phase:Phase;step:number;
  launchSteps:number;fuel:number;lastAction:number[];outcome:string;
  landingXError:number|null;landingZError:number|null;touchdownSpeed:number|null;touchdownTilt:number|null;
};

const C={
  dt:physicsConfig.dt_seconds,maxSteps:physicsConfig.max_episode_steps,maxLaunchSteps:physicsConfig.max_launch_steps,
  gravity:physicsConfig.gravity_mps2,mass:physicsConfig.vehicle_mass_kg,massJitter:physicsConfig.vehicle_mass_jitter_kg,
  chassisLength:physicsConfig.chassis_half_length_m,chassisHeight:physicsConfig.chassis_half_height_m,
  chassisWidth:physicsConfig.chassis_half_width_m,wheelRadius:physicsConfig.wheel_radius_m,
  wheelX:physicsConfig.wheel_x_offset_m,wheelY:physicsConfig.wheel_y_offset_m,wheelZ:physicsConfig.wheel_z_offset_m,
  initialX:physicsConfig.initial_vehicle_x_m,rampStartX:physicsConfig.ramp_start_x_m,
  rampRadius:physicsConfig.ramp_radius_m,rampRadiusJitter:physicsConfig.ramp_radius_jitter_m,
  rampVerticalLength:physicsConfig.ramp_vertical_stem_length_m,rampTopRadius:physicsConfig.ramp_top_curve_radius_m,
  rampTopAngleDeg:physicsConfig.ramp_top_curve_angle_deg,
  rampClearance:physicsConfig.ramp_vehicle_clearance_m,rampSegments:physicsConfig.ramp_segments,
  rampWidth:physicsConfig.ramp_half_width_m,rampThickness:physicsConfig.ramp_thickness_m,rampFriction:physicsConfig.ramp_friction,
  rampFrictionJitter:physicsConfig.ramp_friction_jitter,groundFriction:physicsConfig.ground_friction,
  tireFriction:physicsConfig.tire_friction,airDensity:physicsConfig.air_density_kg_m3,
  dragArea:physicsConfig.drag_coefficient_area_m2,launchForce:physicsConfig.launch_force_n,
  launchJitter:physicsConfig.launch_force_jitter_n,windMax:physicsConfig.wind_force_max_n,
  landingX:physicsConfig.landing_center_x_m,landingXJitter:physicsConfig.landing_center_x_jitter_m,
  landingZJitter:physicsConfig.landing_center_z_jitter_m,padHalfLength:physicsConfig.landing_pad_half_length_m,
  padHalfWidth:physicsConfig.landing_pad_half_width_m,forwardMax:physicsConfig.forward_thrust_max_n,
  verticalMax:physicsConfig.vertical_thrust_max_n,lateralMax:physicsConfig.lateral_thrust_max_n,
  pitchMax:physicsConfig.pitch_thruster_max_n,rollMax:physicsConfig.roll_thruster_max_n,
  yawMax:physicsConfig.yaw_thruster_max_n,nozzleArm:physicsConfig.nozzle_arm_m,
  initialFuel:physicsConfig.initial_fuel_fraction,fuelForward:physicsConfig.fuel_forward_per_second,
  fuelVertical:physicsConfig.fuel_vertical_per_second,fuelLateral:physicsConfig.fuel_lateral_per_second,
  fuelAttitude:physicsConfig.fuel_attitude_per_second,safeHorizontal:physicsConfig.safe_max_horizontal_speed_mps,
  safeVertical:physicsConfig.safe_max_vertical_speed_mps,safeTilt:physicsConfig.safe_max_tilt_rad,
  safeSpin:physicsConfig.safe_max_spin_radps,touchHeight:physicsConfig.touchdown_center_height_m,
  worldX:physicsConfig.world_limit_x_m,worldY:physicsConfig.world_limit_y_m,worldZ:physicsConfig.world_limit_z_m,
  gamma:physicsConfig.training_gamma,crashPenalty:physicsConfig.crash_penalty,
  timeoutPenalty:physicsConfig.timeout_penalty,safeBonus:physicsConfig.safe_landing_bonus,
  flawedTilt:physicsConfig.flawed_upright_tilt_rad,flawedSpeed:physicsConfig.flawed_slow_speed_mps,
  flawedBonus:physicsConfig.flawed_bonus_per_step,shapingScale:physicsConfig.shaping_scale,
  timeCost:physicsConfig.time_cost_per_step,potentialDistance:physicsConfig.potential_distance_weight,
  potentialSpeed:physicsConfig.potential_speed_weight,potentialTilt:physicsConfig.potential_tilt_weight,
  potentialSpin:physicsConfig.potential_spin_weight,potentialFuel:physicsConfig.potential_fuel_weight,
};
const ACTION_FIELDS=['ramp_drive','forward_thrust','vertical_thrust','lateral_thrust','pitch_thrusters','roll_thrusters','yaw_thrusters'];
const TELEMETRY_FIELDS=['step','x','y','z','vx','vy','vz','rotation_qx','rotation_qy','rotation_qz','rotation_qw',
  'omega_x','omega_y','omega_z','fuel',...ACTION_FIELDS,'landing_center_x','landing_center_z','ramp_radius_m',
  'ramp_start_x','ramp_end_x','ramp_top_y','elapsed_seconds'];
const COLUMNS=[...TELEMETRY_FIELDS,'native_total','shaped_total'];

export async function initializeRapier(){await RAPIER.init();}

export class Random {
  private state:number;
  constructor(seed:number){this.state=(seed>>>0)||0x9e3779b9;}
  next(){let x=this.state;x^=x<<13;x^=x>>>17;x^=x<<5;this.state=x>>>0;return this.state/4294967296;}
  uniform(min:number,max:number){return min+(max-min)*this.next();}
  normal(){const u=Math.max(this.next(),1e-12),v=this.next();return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v);}
  int(max:number){return Math.floor(this.next()*max);}
}

function zeros(n:number){return Array.from({length:n},()=>0);}
function layer(input:number,output:number,rng:Random,small=false):DenseLayer{
  const scale=small?0.01:Math.sqrt(2/(input+output));
  return {input,output,weights:Array.from({length:input*output},()=>rng.uniform(-scale,scale)),bias:zeros(output),
    mw:zeros(input*output),vw:zeros(input*output),mb:zeros(output),vb:zeros(output)};
}
function network(rng:Random,input:number,output:number,tanhOutput:boolean,smallOutput:boolean):Network{
  const first=layer(input,24,rng),second=layer(24,output,rng,smallOutput);
  return {layers:[first,second],tanhOutput,step:0};
}
function forward(net:Network,input:number[]):Forward{
  const activations=[input];let x=input;
  for(let li=0;li<net.layers.length;li++){
    const l=net.layers[li],y=zeros(l.output);
    for(let j=0;j<l.output;j++){let sum=l.bias[j];for(let i=0;i<l.input;i++)sum+=l.weights[j*l.input+i]*x[i];
      y[j]=(li===net.layers.length-1&&!net.tanhOutput)?sum:Math.tanh(sum);}
    activations.push(y);x=y;
  }
  return {output:x,activations};
}
function emptyGrad(net:Network){return net.layers.map(l=>({weights:zeros(l.weights.length),bias:zeros(l.bias.length)}));}
function backward(net:Network,cache:Forward,gradient:number[],acc:Array<{weights:number[];bias:number[]}>){
  let delta=gradient;
  for(let li=net.layers.length-1;li>=0;li--){const l=net.layers[li],input=cache.activations[li],output=cache.activations[li+1],g=acc[li];
    const dz=delta.map((v,j)=>v*((li===net.layers.length-1&&!net.tanhOutput)?1:1-output[j]*output[j]));
    for(let j=0;j<l.output;j++){g.bias[j]+=dz[j];for(let i=0;i<l.input;i++)g.weights[j*l.input+i]+=dz[j]*input[i];}
    if(li>0){const previous=zeros(l.input);for(let i=0;i<l.input;i++)for(let j=0;j<l.output;j++)previous[i]+=l.weights[j*l.input+i]*dz[j];delta=previous;}
  }
}
function adam(net:Network,grads:Array<{weights:number[];bias:number[]}>,rate:number){
  net.step++;const b1=.9,b2=.999,eps=1e-8,c1=1-b1**net.step,c2=1-b2**net.step;
  net.layers.forEach((l,i)=>{const update=(p:number[],m:number[],v:number[],g:number[])=>{for(let k=0;k<p.length;k++){
    m[k]=b1*m[k]+(1-b1)*g[k];v[k]=b2*v[k]+(1-b2)*g[k]*g[k];p[k]-=rate*(m[k]/c1)/(Math.sqrt(v[k]/c2)+eps);}};
    update(l.weights,l.mw,l.vw,grads[i].weights);update(l.bias,l.mb,l.vb,grads[i].bias);});
}
function logProbability(action:number[],mean:number[]){let sum=0;for(let i=0;i<action.length;i++){const d=(action[i]-mean[i])/0.4;sum+=-.5*d*d-Math.log(.4*Math.sqrt(2*Math.PI));}return sum;}
function value(net:Network,obs:number[]){return forward(net,obs).output[0];}

export class MiniPPO {
  readonly actor:Network;readonly critic:Network;private updates=0;
  constructor(private rng:Random){this.actor=network(rng,16,7,true,true);this.critic=network(rng,16,1,false,false);}
  sample(obs:number[]){const mean=forward(this.actor,obs).output,action=mean.map(v=>v+.4*this.rng.normal());
    return {action,mean,logProb:logProbability(action,mean),value:value(this.critic,obs)};}
  update(trajectory:Transition[]){
    if(!trajectory.length)return;
    const advantages=zeros(trajectory.length),returns=zeros(trajectory.length);let gae=0,nextValue=0;
    for(let i=trajectory.length-1;i>=0;i--){const t=trajectory[i],delta=t.reward+C.gamma*nextValue-t.value;gae=delta+C.gamma*.95*gae;
      advantages[i]=gae;returns[i]=gae+t.value;nextValue=t.value;}
    const mean=advantages.reduce((a,b)=>a+b,0)/advantages.length;
    const sd=Math.sqrt(advantages.reduce((a,v)=>a+(v-mean)**2,0)/advantages.length+1e-8);
    const normalized=advantages.map(v=>(v-mean)/sd),indices=trajectory.map((_,i)=>i);
    for(let epoch=0;epoch<4;epoch++){for(let i=indices.length-1;i>0;i--){const j=this.rng.int(i+1);[indices[i],indices[j]]=[indices[j],indices[i]];}
      for(let start=0;start<indices.length;start+=64){const batch=indices.slice(start,start+64),actorGrads=emptyGrad(this.actor),criticGrads=emptyGrad(this.critic);
        for(const index of batch){const t=trajectory[index],adv=normalized[index],current=forward(this.actor,t.observation),newLog=logProbability(t.action,current.output);
          const ratio=Math.exp(Math.max(-20,Math.min(20,newLog-t.oldLogProb))),saturated=(adv>0&&ratio>1.2)||(adv<0&&ratio<.8);
          if(!saturated){const coeff=-ratio*adv/batch.length,dMean=current.output.map((m,k)=>coeff*(t.action[k]-m)/(.4*.4));backward(this.actor,current,dMean,actorGrads);}
          const criticCache=forward(this.critic,t.observation),error=criticCache.output[0]-returns[index];backward(this.critic,criticCache,[error/batch.length],criticGrads);}
        adam(this.actor,actorGrads,.0003);adam(this.critic,criticGrads,.001);
      }
    }
    this.updates++;
  }
  export(variant:RewardVariant,seed:number){return {format:'landing-lab-browser-mini-ppo-v4',description:'Client-side PPO actor and value networks for the Rapier3D rigid-body vehicle task; this is not an SB3 checkpoint.',reward_variant:variant,seed,simulator_version:6,policy_updates:this.updates,action_std:.4,
    actor:serialNetwork(this.actor),critic:serialNetwork(this.critic)};}
}
function serialNetwork(net:Network){return {activation_hidden:'tanh',activation_output:net.tanhOutput?'tanh':'linear',layers:net.layers.map(l=>({input:l.input,output:l.output,weights:l.weights,bias:l.bias}))};}

function addGround(world:RAPIER.World){const body=world.createRigidBody(RAPIER.RigidBodyDesc.fixed());
  world.createCollider(RAPIER.ColliderDesc.cuboid(120,.5,30).setTranslation(0,-.5,0).setFriction(C.groundFriction),body);}
function addRamp(world:RAPIER.World,radius:number,friction:number){
  const body=world.createRigidBody(RAPIER.RigidBodyDesc.fixed());
  const geometry=buildRampGeometry({startX:C.rampStartX,radius,clearance:C.rampClearance,verticalLength:C.rampVerticalLength,
    topRadius:C.rampTopRadius,topAngleDeg:C.rampTopAngleDeg,halfWidth:C.rampWidth,thickness:C.rampThickness,segments:C.rampSegments});
  const collider=RAPIER.ColliderDesc.trimesh(new Float32Array(geometry.vertices),new Uint32Array(geometry.indices),RAPIER.TriMeshFlags.FIX_INTERNAL_EDGES)
    .setFriction(friction);
  world.createCollider(collider,body);
  return {start:C.rampStartX,end:geometry.endX,top:geometry.topY,radius};
}
function addCar(world:RAPIER.World,rng:Random){const body=world.createRigidBody(RAPIER.RigidBodyDesc.dynamic()
    .setTranslation(C.initialX,.78,rng.uniform(-.25,.25)).setCanSleep(false).setCcdEnabled(true)
    .setLinearDamping(.015).setAngularDamping(.08));
  world.createCollider(RAPIER.ColliderDesc.cuboid(C.chassisLength,C.chassisHeight,C.chassisWidth)
    .setMass(C.mass+rng.uniform(-C.massJitter,C.massJitter)-60).setFriction(.65).setRestitution(.05),body);
  for(const x of [-C.wheelX,C.wheelX])for(const z of [-C.wheelZ,C.wheelZ])
    world.createCollider(RAPIER.ColliderDesc.ball(C.wheelRadius).setTranslation(x,C.wheelY,z)
      .setMass(15).setFriction(C.tireFriction).setFrictionCombineRule(RAPIER.CoefficientCombineRule.Min)
      .setRestitution(.02),body);
  return body;}
function makeScenario(rng:Random):Scenario{
  const world=new RAPIER.World({x:0,y:-C.gravity,z:0});world.timestep=C.dt;world.numSolverIterations=12;world.maxCcdSubsteps=4;
  const target={x:C.landingX+rng.uniform(-C.landingXJitter,C.landingXJitter),y:0,z:rng.uniform(-C.landingZJitter,C.landingZJitter)};
  const radius=C.rampRadius+rng.uniform(-C.rampRadiusJitter,C.rampRadiusJitter);
  const ramp=addRamp(world,radius,C.rampFriction+rng.uniform(-C.rampFrictionJitter,C.rampFrictionJitter));
  addGround(world);
  return {world,body:addCar(world,rng),target,wind:{x:rng.uniform(-C.windMax,C.windMax),y:0,z:rng.uniform(-C.windMax,C.windMax)},
    launchForce:C.launchForce+rng.uniform(-C.launchJitter,C.launchJitter),rampRadius:ramp.radius,rampStartX:ramp.start,rampEndX:ramp.end,rampTopY:ramp.top,
    phase:'launch',step:0,launchSteps:0,fuel:C.initialFuel,lastAction:[0,0,0,0,0,0,0],outcome:'In progress',
    landingXError:null,landingZError:null,touchdownSpeed:null,touchdownTilt:null};}
function vec(v:Vec3):number[]{return [v.x,v.y,v.z];}
function rotate(v:Vec3,q:Quat):Vec3{const u={x:q.x,y:q.y,z:q.z};
  const cross1={x:u.y*v.z-u.z*v.y,y:u.z*v.x-u.x*v.z,z:u.x*v.y-u.y*v.x};
  const cross2={x:u.y*cross1.z-u.z*cross1.y,y:u.z*cross1.x-u.x*cross1.z,z:u.x*cross1.y-u.y*cross1.x};
  return {x:v.x+2*(cross2.x+q.w*cross1.x),y:v.y+2*(cross2.y+q.w*cross1.y),z:v.z+2*(cross2.z+q.w*cross1.z)};}
function tilt(q:Quat){return Math.acos(Math.max(-1,Math.min(1,rotate({x:0,y:1,z:0},q).y)));}
function observation(s:Scenario){const pos=s.body.translation(),vel=s.body.linvel(),omega=s.body.angvel(),q=s.body.rotation();
  return [Math.max(-1,Math.min(1,(s.target.x-pos.x)/45)),Math.max(-1,Math.min(1,(s.target.y-pos.y)/20)),Math.max(-1,Math.min(1,(s.target.z-pos.z)/20)),
    Math.max(-1,Math.min(1,vel.x/35)),Math.max(-1,Math.min(1,vel.y/35)),Math.max(-1,Math.min(1,vel.z/35)),
    q.x,q.y,q.z,q.w,Math.max(-1,Math.min(1,omega.x/8)),Math.max(-1,Math.min(1,omega.y/8)),Math.max(-1,Math.min(1,omega.z/8)),
    Math.max(0,Math.min(1,s.fuel)),s.phase==='flight'?1:0,Math.max(0,Math.min(1,s.step/C.maxSteps))];}
function telemetry(s:Scenario){const pos=s.body.translation(),vel=s.body.linvel(),omega=s.body.angvel(),q=s.body.rotation();
  return [s.step,pos.x,pos.y,pos.z,vel.x,vel.y,vel.z,q.x,q.y,q.z,q.w,omega.x,omega.y,omega.z,s.fuel,
    ...s.lastAction,s.target.x,s.target.z,s.rampRadius,s.rampStartX,s.rampEndX,s.rampTopY,s.step*C.dt];}
function potential(s:Scenario){const pos=s.body.translation(),vel=s.body.linvel(),omega=s.body.angvel(),dist=Math.hypot(pos.x-s.target.x,pos.y-s.target.y,pos.z-s.target.z);
  return -C.potentialDistance*dist/45-C.potentialSpeed*Math.hypot(vel.x,vel.y,vel.z)/35-
    C.potentialTilt*tilt(s.body.rotation())/Math.PI-C.potentialSpin*Math.hypot(omega.x,omega.y,omega.z)/8+C.potentialFuel*s.fuel;}
function initialForce(s:Scenario,control:number){const pos=s.body.translation(),q=s.body.rotation(),throttle=(control+1)/2,
  force=rotate({x:s.launchForce*throttle,y:0,z:0},q);
  const pointOffset=rotate({x:-1.45,y:-.12,z:0},q);s.body.resetForces(true);s.body.resetTorques(true);
  s.body.addForceAtPoint(force,{x:pos.x+pointOffset.x,y:pos.y+pointOffset.y,z:pos.z+pointOffset.z},true);s.body.addForce(s.wind,true);}
function applyThrusters(s:Scenario,control:number[]){const pos=s.body.translation(),q=s.body.rotation(),arm=C.nozzleArm;
  const [forward,verticalCommand,lateral,pitch,roll,yaw]=control.map(v=>Math.max(-1,Math.min(1,v))),lift=(verticalCommand+1)/2;
  s.body.resetForces(true);s.body.resetTorques(true);
  const nozzles:Array<[Vec3,Vec3]>=[
    [{x:C.forwardMax*forward,y:0,z:0},{x:-arm,y:0,z:0}],
    [{x:0,y:C.verticalMax*lift,z:0},{x:0,y:0,z:0}],
    [{x:0,y:0,z:C.lateralMax*lateral},{x:0,y:0,z:-arm}],
    [{x:0,y:C.pitchMax*pitch,z:0},{x:arm,y:0,z:0}],
    [{x:0,y:C.rollMax*roll,z:0},{x:0,y:0,z:arm}],
    [{x:C.yawMax*yaw,y:0,z:0},{x:0,y:0,z:arm}],
  ];
  const rate=C.fuelForward*Math.abs(forward)+C.fuelVertical*lift+C.fuelLateral*Math.abs(lateral)+
    C.fuelAttitude*(Math.abs(pitch)+Math.abs(roll)+Math.abs(yaw))/3;
  if(s.fuel<=0)return 0;
  for(const [localForce,localPoint] of nozzles){const force=rotate(localForce,q),offset=rotate(localPoint,q);
    s.body.addForceAtPoint(force,{x:pos.x+offset.x,y:pos.y+offset.y,z:pos.z+offset.z},true);}
  return Math.min(s.fuel,rate*C.dt);}
function closeScenario(s:Scenario){s.world.free();}

function rollout(agent:MiniPPO,variant:RewardVariant,rng:Random,seed:number,episode:number):{result:BrowserEpisode;transitions:Transition[]}{
  const s=makeScenario(rng),transitions:Transition[]=[],rows:number[][]=[telemetry(s)],phases:string[]=['launch'];
  const nativeRewards:number[]=[],shapedRewards:number[]=[];let nativeReturn=0,shapedReturn=0,lastPotential=potential(s),terminated=false,truncated=false;
  while(!terminated&&!truncated&&s.step<C.maxSteps){
    const policyControlled=true,obs=observation(s),sample=agent.sample(obs),control=sample.action.map(v=>Math.max(-1,Math.min(1,v))),pos=s.body.translation();
    const previousDistance=Math.hypot(pos.x-s.target.x,pos.y-s.target.y,pos.z-s.target.z);let base=0,fuelUsed=0,applied=[0,0,0,0,0,0,0];
    s.step++;
    if(s.phase==='launch'){s.launchSteps++;applied[0]=control[0];initialForce(s,control[0]);}
    else{
      applied=[-1,...control.slice(1)];fuelUsed=applyThrusters(s,control.slice(1));s.body.addForce(s.wind,true);
      const v=s.body.linvel(),speed=Math.hypot(v.x,v.y,v.z),dragScale=-.5*C.airDensity*C.dragArea*speed;
      s.body.addForce({x:dragScale*v.x,y:dragScale*v.y,z:dragScale*v.z},true);s.fuel=Math.max(0,s.fuel-fuelUsed);
    }
    s.world.step();
    const p=s.body.translation(),v=s.body.linvel(),w=s.body.angvel(),q=s.body.rotation(),speed=Math.hypot(v.x,v.y,v.z),attitude=tilt(q);
    base=.08*(previousDistance-Math.hypot(p.x-s.target.x,p.y-s.target.y,p.z-s.target.z))-
      .003*speed-.008*attitude-.003*Math.hypot(w.x,w.y,w.z)-.08*fuelUsed;
    if(s.phase==='launch'){
      if(p.y>=s.rampTopY-1.1&&p.x<=s.rampEndX+.8&&v.x<-.5)s.phase='flight';
      else if(s.launchSteps>=C.maxLaunchSteps){s.outcome='Launch failure';base=C.crashPenalty;terminated=true;}
    }else if(p.y<=C.touchHeight&&v.y<=0){
      s.landingXError=Math.abs(p.x-s.target.x);s.landingZError=Math.abs(p.z-s.target.z);s.touchdownSpeed=speed;s.touchdownTilt=attitude;
      const safe=s.landingXError<=C.padHalfLength&&s.landingZError<=C.padHalfWidth&&Math.hypot(v.x,v.z)<=C.safeHorizontal&&
        Math.abs(v.y)<=C.safeVertical&&attitude<=C.safeTilt&&Math.hypot(w.x,w.y,w.z)<=C.safeSpin;
      s.outcome=safe?'Safe landing':'Crash';base+=safe?C.safeBonus:C.crashPenalty;terminated=true;
    }
    if(!terminated&&!truncated&&(Math.abs(p.x)>C.worldX||p.y< -10||p.y>C.worldY||Math.abs(p.z)>C.worldZ)){
      s.outcome='Crash';base+=C.crashPenalty;terminated=true;
    }
    if(s.step>=C.maxSteps&&!terminated){s.outcome='Timeout';base+=C.timeoutPenalty;truncated=true;}
    let addition=0;const nextPotential=terminated||truncated?0:potential(s);
    if(variant==='upright_bonus'&&policyControlled&&!terminated&&!truncated&&attitude<C.flawedTilt&&speed<C.flawedSpeed)addition=C.flawedBonus;
    else if(variant==='potential_shaping'&&policyControlled)addition=C.shapingScale*(C.gamma*nextPotential-lastPotential)-C.timeCost;
    lastPotential=policyControlled?nextPotential:potential(s);
    const shaped=base+addition;nativeReturn+=base;shapedReturn+=shaped;nativeRewards.push(base);shapedRewards.push(shaped);
    transitions.push({observation:obs,action:sample.action,oldLogProb:sample.logProb,reward:shaped,value:sample.value});
    s.lastAction=applied;rows.push(telemetry(s));phases.push(s.phase);
  }
  const position=s.body.translation(),velocity=s.body.linvel(),angular=s.body.angvel(),q=s.body.rotation(),safe=s.outcome==='Safe landing';
  const flight:SavedRoadster={schema:4,simulator_version:6,columns:COLUMNS,rows:rows.map(row=>row.map(v=>Number(v.toFixed(6)))),phases,
    ramp_radius_m:s.rampRadius,ramp_start_x:s.rampStartX,ramp_end_x:s.rampEndX,ramp_top_y:s.rampTopY,
    ramp_vertical_stem_length_m:C.rampVerticalLength,ramp_top_curve_radius_m:C.rampTopRadius,ramp_top_curve_angle_deg:C.rampTopAngleDeg,
    landing_zone_x:[s.target.x-C.padHalfLength,s.target.x+C.padHalfLength],landing_zone_z:[s.target.z-C.padHalfWidth,s.target.z+C.padHalfWidth],
    outcome:s.outcome,native_return:nativeReturn,shaped_return:shapedReturn,steps:rows.length-1,
    metrics:{landing_x_error:s.landingXError,landing_z_error:s.landingZError,touchdown_speed:s.touchdownSpeed,touchdown_tilt:s.touchdownTilt,
      gas_remaining:s.fuel,elapsed_seconds:s.step*C.dt,final_x:position.x,final_y:position.y,final_z:position.z,
      final_vx:velocity.x,final_vy:velocity.y,final_vz:velocity.z,final_omega:Math.hypot(angular.x,angular.y,angular.z),
      final_rotation:[q.x,q.y,q.z,q.w],landing_zone_x:[s.target.x-C.padHalfLength,s.target.x+C.padHalfLength],
      landing_zone_z:[s.target.z-C.padHalfWidth,s.target.z+C.padHalfWidth]},
    source:{kind:'browser_training',variant,seed,episode},
  };
  closeScenario(s);
  return {result:{episode,outcome:s.outcome,nativeReturn,shapedReturn,length:rows.length-1,safe,flight},transitions};
}

export async function trainBrowserPPO(options:{variant:RewardVariant;seed:number;episodes:number;shouldStop:()=>boolean;onEpisode:(episode:BrowserEpisode,policyUpdates:number)=>void}):Promise<{stopped:boolean;policy:unknown}>{
  const rng=new Random(options.seed),agent=new MiniPPO(rng);let completed=0;
  for(let episode=1;episode<=options.episodes;episode++){
    if(options.shouldStop())break;
    const rolloutResult=rollout(agent,options.variant,rng,options.seed,episode);agent.update(rolloutResult.transitions);completed++;
    options.onEpisode(rolloutResult.result,completed);await new Promise<void>(resolve=>setTimeout(resolve,0));
  }
  return {stopped:options.shouldStop(),policy:agent.export(options.variant,options.seed)};
}
