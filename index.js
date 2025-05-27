// data.js
import fs from 'fs';
import csv from 'csv-parser';
import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
import * as d3 from 'd3';
import { createCanvas } from 'canvas';
import { KMeans } from 'ml-kmeans';
import { silhouetteScore } from 'ml-silhouette';
import { ANOVA } from 'jstat';

// Helper function to read CSV file
async function readCSV(filePath) {
  return new Promise((resolve, reject) => {
    const results = [];
    fs.createReadStream(filePath)
      .pipe(csv())
      .on('data', (data) => results.push(data))
      .on('end', () => {
        console.log(`Data berhasil dimuat. Jumlah baris: ${results.length}`);
        resolve(results);
      })
      .on('error', (error) => {
        console.error("Terjadi kesalahan saat membaca file:", error);
        reject(error);
      });
  });
}

// StandardScaler implementation
class StandardScaler {
  constructor() {
    this.mean = null;
    this.std = null;
  }

  fit(data) {
    this.mean = data.reduce((acc, val) => acc.map((v, i) => v + val[i]), 
                          new Array(data[0].length).fill(0))
                  .map(sum => sum / data.length);
    
    this.std = data.reduce((acc, val) => acc.map((v, i) => v + Math.pow(val[i] - this.mean[i], 2)), 
                         new Array(data[0].length).fill(0))
                 .map(sum => Math.sqrt(sum / data.length))
                 .map(v => v === 0 ? 1 : v); // avoid division by zero
  }

  transform(data) {
    return data.map(row => 
      row.map((val, i) => (val - this.mean[i]) / this.std[i])
    );
  }

  fit_transform(data) {
    this.fit(data);
    return this.transform(data);
  }
}

// Main analysis function
async function analyzeData() {
  let df;
  try {
    df = await readCSV('data_siswa.csv');
  } catch (error) {
    console.error("File tidak ditemukan. Pastikan file 'data_siswa.csv' ada di direktori yang sama.");
    process.exit(1);
  }

  // Clean column names
  df = df.map(row => {
    const cleanRow = {};
    for (const key in row) {
      cleanRow[key.trim()] = row[key];
    }
    return cleanRow;
  });

  // Select relevant columns
  const relevantCols = ['Durasi', 'JumlahStress', 'JumlahCemas'];
  df = df.map(row => {
    const filteredRow = {};
    relevantCols.forEach(col => {
      filteredRow[col] = row[col];
    });
    return filteredRow;
  });

  // Remove rows with missing values
  df = df.filter(row => 
    row.Durasi && row.JumlahStress && row.JumlahCemas && 
    row.JumlahStress !== '' && row.JumlahCemas !== ''
  );

  // Map duration to numeric values
  const durasiMapping = {
    'Kurang dari 2 jam': 1,
    '2-4 jam': 3,
    'Lebih dari 4 jam': 5
  };

  df = df.map(row => {
    return {
      ...row,
      Durasi_numerik: durasiMapping[row.Durasi],
      JumlahStress: parseFloat(row.JumlahStress),
      JumlahCemas: parseFloat(row.JumlahCemas)
    };
  }).filter(row => !isNaN(row.JumlahStress) && !isNaN(row.JumlahCemas));

  // Prepare data for clustering
  const features = df.map(row => [
    row.Durasi_numerik,
    row.JumlahStress,
    row.JumlahCemas
  ]);

  // Standardize the data
  const scaler = new StandardScaler();
  const scaledData = scaler.fit_transform(features);

  // Elbow method to determine optimal number of clusters
  const wcss = [];
  for (let i = 1; i <= 5; i++) {
    const kmeans = new KMeans(scaledData, { k: i, initialization: 'kmeans++' });
    wcss.push(kmeans.computeInformation());
  }

  // Plot elbow method (would be shown in browser with tfjs-vis)
  console.log('WCSS values for elbow method:', wcss);

  // Perform K-means clustering with k=2
  const k = 2;
  const kmeans = new KMeans(scaledData, { k, initialization: 'kmeans++' });
  const clusters = kmeans.predict(scaledData);

  // Add cluster labels to dataframe
  df = df.map((row, index) => ({
    ...row,
    cluster: clusters[index]
  }));

  // Calculate silhouette score
  const silhouette_avg = silhouetteScore(scaledData, clusters);
  console.log(`Silhouette Score: ${silhouette_avg.toFixed(2)}`);

  // Cluster analysis
  const clusterGroups = {};
  df.forEach(row => {
    if (!clusterGroups[row.cluster]) {
      clusterGroups[row.cluster] = [];
    }
    clusterGroups[row.cluster].push(row);
  });

  const clusterAnalysis = {};
  for (const [cluster, rows] of Object.entries(clusterGroups)) {
    clusterAnalysis[cluster] = {
      Durasi_numerik: {
        mean: d3.mean(rows, d => d.Durasi_numerik),
        min: d3.min(rows, d => d.Durasi_numerik),
        max: d3.max(rows, d => d.Durasi_numerik)
      },
      JumlahStress: {
        mean: d3.mean(rows, d => d.JumlahStress),
        min: d3.min(rows, d => d.JumlahStress),
        max: d3.max(rows, d => d.JumlahStress)
      },
      JumlahCemas: {
        mean: d3.mean(rows, d => d.JumlahCemas),
        min: d3.min(rows, d => d.JumlahCemas),
        max: d3.max(rows, d => d.JumlahCemas)
      },
      count: rows.length
    };
  }

  console.log("\nAnalisis Cluster:");
  console.log(clusterAnalysis);

  // Duration category comparison
  df = df.map(row => ({
    ...row,
    durasi_kategori: row.Durasi === '2-4 jam' ? '2-4 jam' : 
                    (row.Durasi === 'Lebih dari 4 jam' ? '>4 jam' : '<2 jam')
  }));

  const durationGroups = {};
  df.forEach(row => {
    if (!durationGroups[row.durasi_kategori]) {
      durationGroups[row.durasi_kategori] = [];
    }
    durationGroups[row.durasi_kategori].push(row);
  });

  const comparison = {};
  for (const [duration, rows] of Object.entries(durationGroups)) {
    comparison[duration] = {
      JumlahStress: d3.mean(rows, d => d.JumlahStress),
      JumlahCemas: d3.mean(rows, d => d.JumlahCemas)
    };
  }

  console.log("\nPerbandingan Durasi Penggunaan Gadget:");
  console.log(comparison);

  // Statistics by duration
  const stats = {};
  for (const [duration, rows] of Object.entries(durationGroups)) {
    stats[duration] = {
      JumlahStress: {
        mean: d3.mean(rows, d => d.JumlahStress),
        median: d3.median(rows, d => d.JumlahStress),
        std: d3.deviation(rows, d => d.JumlahStress)
      },
      JumlahCemas: {
        mean: d3.mean(rows, d => d.JumlahCemas),
        median: d3.median(rows, d => d.JumlahCemas),
        std: d3.deviation(rows, d => d.JumlahCemas)
      }
    };
  }

  console.log("\nRata-rata Tingkat Stres dan Cemas berdasarkan Durasi:");
  console.log(stats);

  // ANOVA tests
  const stressGroups = [
    df.filter(d => d.durasi_kategori === '<2 jam').map(d => d.JumlahStress),
    df.filter(d => d.durasi_kategori === '2-4 jam').map(d => d.JumlahStress),
    df.filter(d => d.durasi_kategori === '>4 jam').map(d => d.JumlahStress)
  ];

  const anovaStress = ANOVA(...stressGroups);
  console.log("\nUji Signifikansi Perbedaan Tingkat Stres:");
  console.log(`F-value: ${anovaStress.F.toFixed(2)}, p-value: ${anovaStress.p.toFixed(4)}`);

  const anxietyGroups = [
    df.filter(d => d.durasi_kategori === '<2 jam').map(d => d.JumlahCemas),
    df.filter(d => d.durasi_kategori === '2-4 jam').map(d => d.JumlahCemas),
    df.filter(d => d.durasi_kategori === '>4 jam').map(d => d.JumlahCemas)
  ];

  const anovaAnxiety = ANOVA(...anxietyGroups);
  console.log("\nUji Signifikansi Perbedaan Tingkat Cemas:");
  console.log(`F-value: ${anovaAnxiety.F.toFixed(2)}, p-value: ${anovaAnxiety.p.toFixed(4)}`);
}

// Run the analysis
analyzeData().catch(console.error);

export { analyzeData };
