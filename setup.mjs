import { mkdir, writeFile } from 'node:fs/promises';
const files = {
  'vendor/tf.cjs': 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js',
};
for (const split of ['train','test']) {
  files[`data/${split}.ts`] = `https://raw.githubusercontent.com/aeon-toolkit/aeon/main/aeon/datasets/data/GunPoint/GunPoint_${split.toUpperCase()}.ts`;
}
for (const [file,url] of Object.entries(files)) {
  const response = await fetch(url);
  if (!response.ok) throw Error(url + ' ' + response.status);
  await mkdir(file.split('/')[0], {recursive:true});
  await writeFile(file, Buffer.from(await response.arrayBuffer()));
}
