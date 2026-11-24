const fs = require('node:fs');
const {tf,load,model,tokens} = require('./model.cjs');
const write = (name,value)=>fs.writeFileSync('results/'+name+'.json',JSON.stringify(value,null,2));
function batch(rows) {
  return [tf.tensor3d(rows.map(r=>r.x)),tf.tensor1d(rows.map(r=>r.y),'int32')];
}
function measure(net,x,y) {
  return tf.tidy(()=>{
    const scores = net.logits(x);
    return {loss:tf.losses.softmaxCrossEntropy(tf.oneHot(y,2),scores).dataSync()[0],
      predictions:Array.from(scores.argMax(1).dataSync())};
  });
}
function train() {
  const rows = load('train');
  const counts=[0,0], fit=[],val=[];
  for (const row of rows) (counts[row.y]++ % 5 === 0 ? val : fit).push(row);
  const [x,y]=batch(fit), [vx,vy]=batch(val);
  const net=model(), optimizer=tf.train.adam(0.008), history=[];
  let best=Infinity;
  for(let epoch=1;epoch<=100;epoch++) {
    optimizer.minimize(()=>tf.losses.softmaxCrossEntropy(tf.oneHot(y,2),net.logits(x)));
    const training=measure(net,x,y), validation=measure(net,vx,vy);
    history.push({epoch,training:training.loss,validation:validation.loss});
    if(validation.loss < best) {
      best=validation.loss;
      write('checkpoint',{epoch,weights:net.snapshot()});
    }
  }
  write('history',history);
}
function evaluate(net) {
  const rows=load('test'), [x,y]=batch(rows);
  const {predictions}=measure(net,x,y), matrix=[[0,0],[0,0]];
  rows.forEach((row,i)=>matrix[row.y][predictions[i]]++);
  const classes=matrix.map((row,i)=>{
    const precision=row[i]/(matrix[0][i]+matrix[1][i] || 1), recall=row[i]/(row[0]+row[1]);
    return {label:i+1,precision,recall,f1:2*precision*recall/(precision+recall || 1)};
  });
  const metrics={samples:rows.length,accuracy:(matrix[0][0]+matrix[1][1])/rows.length,matrix,classes};
  for(const key of ['precision','recall','f1']) metrics[key]=classes.reduce((s,c)=>s+c[key],0)/2;
  write('metrics',metrics);
  console.log(metrics);
}
const mode=process.argv[2];
if(mode==='train') train();
else {
  const net=model(JSON.parse(fs.readFileSync('results/checkpoint.json')).weights);
  if(mode==='evaluate') evaluate(net);
  else if(mode==='predict') {
    const trace=JSON.parse(fs.readFileSync(process.argv[3]));
    console.log('Archive class',1+net.logits(tf.tensor3d([tokens(trace)])).argMax(1).dataSync()[0]);
  } else throw Error('Use train, evaluate, or predict <trace.json>');
}
