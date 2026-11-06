export type RampGeometryConfig={
  startX:number;radius:number;clearance:number;verticalLength:number;topRadius:number;topAngleDeg:number;
  halfWidth:number;thickness:number;segments:number;
};

export type RampGeometry={
  vertices:number[];indices:number[];endX:number;topY:number;centerline:Array<[number,number,number]>;
};

/**
 * Generate the shared v6 Roadster ramp profile. The centerline rises through a
 * broad quarter turn, a near-vertical straight section, and a short curved
 * shoulder. The physical road is offset by the car clearance and built as a
 * closed triangle mesh for Rapier.
 */
export function buildRampGeometry(config:RampGeometryConfig):RampGeometry{
  const {startX,radius,clearance,verticalLength,topRadius,topAngleDeg,halfWidth,thickness,segments}=config;
  const topAngle=topAngleDeg*Math.PI/180;
  if(segments<32||radius<=0||clearance<0||verticalLength<0||topRadius<=0||topAngle<=0||topAngle>=Math.PI/2||halfWidth<=0||thickness<=0)
    throw new Error('Invalid v6 ramp geometry settings.');
  const baseLength=radius*Math.PI/2,shoulderLength=topRadius*topAngle,totalLength=baseLength+verticalLength+shoulderLength;
  const baseSegments=Math.max(12,Math.round(segments*baseLength/totalLength));
  const stemSegments=Math.max(4,Math.round(segments*verticalLength/totalLength));
  const shoulderSegments=Math.max(8,segments-baseSegments-stemSegments);
  const centerline:Array<[number,number,number]>=[];
  for(let i=0;i<=baseSegments;i++){
    const angle=Math.PI/2*i/baseSegments;
    centerline.push([radius*Math.sin(angle),radius+clearance-radius*Math.cos(angle),angle]);
  }
  for(let i=1;i<=stemSegments;i++){
    const fraction=i/stemSegments;
    centerline.push([radius,radius+clearance+verticalLength*fraction,Math.PI/2]);
  }
  for(let i=1;i<=shoulderSegments;i++){
    const angle=Math.PI/2+topAngle*i/shoulderSegments;
    centerline.push([radius+topRadius*(Math.sin(angle)-1),radius+clearance+verticalLength-topRadius*Math.cos(angle),angle]);
  }
  const vertices:number[]=[];
  for(const [pathX,pathY,tangent] of centerline){
    const nx=-Math.sin(tangent),ny=Math.cos(tangent),x=pathX+startX;
    const innerX=x-clearance*nx,innerY=pathY-clearance*ny;
    const outerX=innerX-thickness*nx,outerY=innerY-thickness*ny;
    vertices.push(innerX,innerY,-halfWidth,innerX,innerY,halfWidth,outerX,outerY,-halfWidth,outerX,outerY,halfWidth);
  }
  const indices:number[]=[];
  for(let i=0;i<centerline.length-1;i++){
    const a=4*i,b=a+4;
    indices.push(a,a+1,b+1,a,b+1,b);
    indices.push(a+2,b+2,b+3,a+2,b+3,a+3);
    indices.push(a+1,b+3,b+1,a+1,a+3,b+3);
    indices.push(a,b,b+2,a,b+2,a+2);
  }
  const last=4*(centerline.length-1);
  indices.push(0,2,3,0,3,1,last,last+1,last+3,last,last+3,last+2);
  const [, ,endAngle]=centerline[centerline.length-1];
  const [pathEndX,pathEndY]=centerline[centerline.length-1];
  return {
    vertices,indices,
    endX:pathEndX+startX+clearance*Math.sin(endAngle),
    topY:pathEndY-clearance*Math.cos(endAngle),
    centerline,
  };
}
