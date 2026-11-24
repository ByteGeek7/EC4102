const fs = require('node:fs');
const tf = require('../vendor/tf.cjs');
const mean = a => a.reduce((s,v)=>s+v,0)/a.length;
function tokens(trace) {
  if (trace.length !== 150 || !trace.every(Number.isFinite)) throw Error('Expected 150 finite samples');
  return Array.from({length:15}, (_,t)=>{
    const a = trace.slice(10*t,10*t+10);
    const d = a.slice(1).map((v,i)=>v-a[i]);
    const roughness = mean(d.slice(1).map((v,i)=>Math.abs(v-d[i])));
    // First two non-DC Fourier magnitudes retain local oscillation information.
    const spectrum = [1,2].map(k => Math.hypot(...['cos','sin'].map(fn =>
      a.reduce((s,v,i)=>s+v*Math[fn](2*Math.PI*k*i/10),0)))) ;
    return [mean(a),10*mean(d),Math.log1p(100*roughness),...spectrum,Math.sin(t/4),Math.cos(t/4)];
  });
}
function load(split) {
  return fs.readFileSync('data/'+split+'.ts','utf8').split(/\r?\n/).filter(l=>/^[-.\d]/.test(l)).map(l=>{
    const [series,label] = l.split(':');
    return {x:tokens(series.split(',').map(Number)), y:Number(label)-1};
  });
}
function model(saved) {
  const params = {};
  const shapes = {embed:[7,12], q:[12,12], k:[12,12], v:[12,12], up:[12,24], down:[24,12], head:[24,2], bias:[2]};
  let seed = 17;
  for (const [name,shape] of Object.entries(shapes))
    params[name] = tf.variable(saved ? tf.tensor(saved[name],shape) : tf.randomNormal(shape,0,0.22,'float32',seed++));
  function linear(x,w) {
    return x.reshape([-1,x.shape[2]]).matMul(w).reshape([x.shape[0],x.shape[1],w.shape[1]]);
  }
  function norm(x) {
    const {mean,variance} = tf.moments(x,-1,true);
    return x.sub(mean).div(variance.add(1e-5).sqrt());
  }
  function logits(x) {
    const h = linear(x,params.embed);
    const weights = tf.softmax(tf.matMul(linear(h,params.q),linear(h,params.k),false,true).div(Math.sqrt(12)));
    const z = norm(h.add(tf.matMul(weights,linear(h,params.v))));
    const out = norm(z.add(linear(linear(z,params.up).relu(),params.down)));
    return tf.concat([out.mean(1),out.max(1)],1).matMul(params.head).add(params.bias);
  }
  return {logits, snapshot:()=>Object.fromEntries(Object.entries(params).map(([k,v])=>[k,v.arraySync()]))};
}
module.exports = {tf,load,model,tokens};
