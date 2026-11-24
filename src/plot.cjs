const fs = require('node:fs');
const read = name => JSON.parse(fs.readFileSync('results/'+name+'.json'));
const metrics = read('metrics'), history = read('history');
const text = (x,y,value,size=18,color='#e5edf5') =>
  `<text x="${x}" y="${y}" font-family="Arial,sans-serif" font-size="${size}" fill="${color}">${value}</text>`;
const peak = Math.max(...history.flatMap(r=>[r.training,r.validation]));
const line = (key,color) => '<polyline fill="none" stroke="'+color+'" stroke-width="3" points="'+history.map((r,i)=>
  (65+420*i/(history.length-1))+','+(245-150*r[key]/peak)).join(' ')+'"/>';
let svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 820 320"><rect width="820" height="320" rx="18" fill="#102539"/>';
svg += text(28,43,'Motion pattern analysis',28)+text(30,76,'Cross-entropy loss')+text(530,76,'Held-out test counts');
svg += text(12,103,peak.toFixed(2),14)+text(42,245,'0',14)+line('training','#59e1c5')+line('validation','#ffbd79');
svg += text(65,270,'1',14)+text(230,270,'Epoch',14)+text(462,270,'100',14)+text(65,300,'Training',16,'#59e1c5')+text(160,300,'Validation',16,'#ffbd79');
metrics.matrix.forEach((row,i)=>row.forEach((n,j)=>{
  const x=602+84*j,y=110+64*i;
  svg += `<rect x="${x}" y="${y}" width="80" height="60" fill="${i===j?'#17695d':'#344d60'}"/>`+text(x+27,y+38,n,22);
}));
svg += text(613,103,'1',14)+text(699,103,'2',14)+text(531,148,'True 1',15)+text(531,211,'True 2',15)+text(609,260,'Predicted class',15);
fs.writeFileSync('results/evaluation.svg',svg+'</svg>');
