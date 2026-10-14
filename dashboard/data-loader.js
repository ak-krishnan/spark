const fs = require('fs');
const path = require('path');

const PROJECT_ROOT = path.join(__dirname, '..');

class DataLoader {
    constructor() {
        this.metrics = null;
        this.projectInfo = null;
        this.predictions = [];
        this.predictionStats = null;
        this.sparkExecution = null;
        this.sourceCode = null;
        this.loadTimestamp = null;
    }

    loadAll() {
        console.log('[DataLoader] Loading project data from real Spark outputs...');
        const start = Date.now();

        this.loadMetrics();
        console.log(`  ✓ Metrics loaded`);

        this.loadPredictions();
        console.log(`  ✓ ${this.predictions.length.toLocaleString()} predictions loaded`);

        this.computePredictionStats();
        console.log(`  ✓ Prediction statistics computed`);

        this.parseSparkExecution();
        console.log(`  ✓ Spark execution logs parsed`);

        this.loadSourceCode();
        console.log(`  ✓ Source code loaded`);

        this.loadTimestamp = new Date().toISOString();
        console.log(`[DataLoader] All data loaded in ${Date.now() - start}ms`);
    }

    // ── Metrics ──────────────────────────────────────────────
    loadMetrics() {
        const filePath = path.join(PROJECT_ROOT, 'results', 'ml_metrics.txt');
        const content = fs.readFileSync(filePath, 'utf-8');
        const lines = content.split('\n');

        const metrics = {};
        const info = {};

        for (const line of lines) {
            const colonIdx = line.indexOf(':');
            if (colonIdx === -1) continue;

            const key = line.substring(0, colonIdx).trim();
            const value = line.substring(colonIdx + 1).trim();

            if (!key || key.startsWith('=')) continue;

            const numValue = parseFloat(value);
            if (['Accuracy', 'Weighted Precision', 'Weighted Recall', 'F1-score', 'ROC-AUC'].includes(key)) {
                metrics[key] = numValue;
            } else if (!isNaN(numValue) && key !== 'Dataset') {
                info[key] = numValue;
            } else {
                info[key] = value;
            }
        }

        this.metrics = metrics;
        this.projectInfo = {
            dataset: info['Dataset'] || 'Amazon Polarity',
            totalReviews: info['Total reviews'] || 3600000,
            trainingReviews: info['Training reviews'] || 2880586,
            testingReviews: info['Testing reviews'] || 719414,
            model: info['Model'] || 'Logistic Regression',
            features: info['Features'] || 'HashingTF (65536 dimensions)',
            wordElements: 275519555,
            partitions: 12,
            sparkVersion: '3.5.1',
            scalaVersion: '2.12.18',
            javaVersion: '17.0.20.1',
            executionMode: 'local[4]',
            executionDate: '2026-10-13T18:03:47',
            hostname: 'mithun-HP-EliteBook-840-G6'
        };
    }

    // ── Predictions ──────────────────────────────────────────
    loadPredictions() {
        const predictionsDir = path.join(PROJECT_ROOT, 'results', 'predictions_csv');
        const files = fs.readdirSync(predictionsDir)
            .filter(f => f.startsWith('part-') && f.endsWith('.csv'))
            .sort();

        this.predictions = [];

        for (const file of files) {
            const content = fs.readFileSync(path.join(predictionsDir, file), 'utf-8');
            const lines = content.split('\n');

            for (let i = 1; i < lines.length; i++) {
                const line = lines[i].trim();
                if (!line) continue;
                const parts = line.split(',');
                if (parts.length < 3) continue;

                this.predictions.push({
                    label: parseFloat(parts[0]),
                    prediction: parseFloat(parts[1]),
                    probability: parseFloat(parts[2])
                });
            }
        }
    }

    // ── Prediction Statistics ────────────────────────────────
    computePredictionStats() {
        const preds = this.predictions;
        const n = preds.length;

        // Confusion matrix
        let tp = 0, tn = 0, fp = 0, fn = 0;
        for (let i = 0; i < n; i++) {
            const actual = preds[i].label;
            const predicted = preds[i].prediction;
            if (actual === 1 && predicted === 1) tp++;
            else if (actual === 0 && predicted === 0) tn++;
            else if (actual === 0 && predicted === 1) fp++;
            else if (actual === 1 && predicted === 0) fn++;
        }

        // Probability distributions (50 bins, separated by actual class)
        const numBins = 50;
        const posHist = new Array(numBins).fill(0);
        const negHist = new Array(numBins).fill(0);
        for (let i = 0; i < n; i++) {
            const bin = Math.min(Math.floor(preds[i].probability * numBins), numBins - 1);
            if (preds[i].label === 1) posHist[bin]++;
            else negHist[bin]++;
        }

        const histogramLabels = [];
        for (let i = 0; i < numBins; i++) {
            histogramLabels.push((i / numBins).toFixed(2));
        }

        // ROC curve (sampled)
        const sorted = preds
            .map((p, idx) => ({ prob: p.probability, label: p.label }))
            .sort((a, b) => b.prob - a.prob);

        const totalPos = tp + fn;
        const totalNeg = tn + fp;
        const rocPoints = [{ fpr: 0, tpr: 0, threshold: 1.0 }];
        let cumulTP = 0, cumulFP = 0;
        const step = Math.max(1, Math.floor(n / 300));

        for (let i = 0; i < sorted.length; i++) {
            if (sorted[i].label === 1) cumulTP++;
            else cumulFP++;

            if (i % step === 0 || i === sorted.length - 1) {
                rocPoints.push({
                    fpr: +(cumulFP / totalNeg).toFixed(5),
                    tpr: +(cumulTP / totalPos).toFixed(5),
                    threshold: +sorted[i].prob.toFixed(5)
                });
            }
        }

        // Threshold analysis
        const thresholdAnalysis = [];
        for (let t = 0.05; t <= 0.95; t += 0.05) {
            let tTP = 0, tFP = 0, tFN = 0, tTN = 0;
            for (let i = 0; i < n; i++) {
                const pred = preds[i].probability >= t ? 1 : 0;
                const actual = preds[i].label;
                if (actual === 1 && pred === 1) tTP++;
                else if (actual === 0 && pred === 1) tFP++;
                else if (actual === 1 && pred === 0) tFN++;
                else tTN++;
            }
            const precision = tTP + tFP > 0 ? tTP / (tTP + tFP) : 0;
            const recall = tTP + tFN > 0 ? tTP / (tTP + tFN) : 0;
            const f1 = precision + recall > 0 ? 2 * precision * recall / (precision + recall) : 0;
            thresholdAnalysis.push({
                threshold: +t.toFixed(2),
                precision: +precision.toFixed(4),
                recall: +recall.toFixed(4),
                f1: +f1.toFixed(4),
                accuracy: +((tTP + tTN) / n).toFixed(4)
            });
        }

        this.predictionStats = {
            total: n,
            confusionMatrix: { tp, tn, fp, fn },
            accuracy: (tp + tn) / n,
            classDistribution: {
                positive: { actual: tp + fn, predicted: tp + fp },
                negative: { actual: tn + fp, predicted: tn + fn }
            },
            probabilityHistogram: {
                labels: histogramLabels,
                positive: posHist,
                negative: negHist
            },
            rocCurve: rocPoints,
            thresholdAnalysis
        };
    }

    // ── Spark Execution Parsing ──────────────────────────────
    parseSparkExecution() {
        const logPath = path.join(PROJECT_ROOT, 'results', 'scala_run_20261013_180347.log');
        const summaryPath = path.join(PROJECT_ROOT, 'results', 'scala_execution_log.txt');

        const rawLog = fs.readFileSync(logPath, 'utf-8');
        const summary = fs.readFileSync(summaryPath, 'utf-8');
        const logLines = rawLog.split('\n');

        // Parse timeline events from log
        const timeline = [];
        const timestampRegex = /^(\d{2}\/\d{2}\/\d{2} \d{2}:\d{2}:\d{2})\s+(WARN|INFO|ERROR)\s+(\w+):\s*(.*)$/;

        for (const line of logLines) {
            const match = line.match(timestampRegex);
            if (match) {
                timeline.push({
                    timestamp: match[1],
                    level: match[2],
                    component: match[3],
                    message: match[4].substring(0, 200)
                });
            }
        }

        // Extract key events for the timeline visualization
        const keyEvents = [];
        let startTime = null, endTime = null;

        for (const entry of timeline) {
            const ts = entry.timestamp;
            if (!startTime) startTime = ts;
            endTime = ts;

            if (entry.message.includes('Running Spark version')) {
                keyEvents.push({ time: ts, event: 'Spark Initialized', detail: 'Spark 3.5.1 started', type: 'start' });
            } else if (entry.message.includes('Submitted application')) {
                keyEvents.push({ time: ts, event: 'Application Submitted', detail: 'AmazonReviewRDD', type: 'info' });
            } else if (entry.message.includes('Starting executor')) {
                keyEvents.push({ time: ts, event: 'Executor Started', detail: 'Driver executor on local node', type: 'info' });
            }
        }

        // Find RDD output events from non-timestamped lines
        for (const line of logLines) {
            if (line.startsWith('RDD created successfully')) {
                keyEvents.push({ time: startTime, event: 'RDD Created', detail: 'CSV loaded into RDD', type: 'success' });
            } else if (line.startsWith('Total reviews:')) {
                keyEvents.push({ time: startTime, event: 'Data Counted', detail: line.trim(), type: 'success' });
            } else if (line.startsWith('Negative review count:')) {
                keyEvents.push({ time: startTime, event: 'Filter Applied', detail: line.trim(), type: 'success' });
            } else if (line.startsWith('Cached label count:')) {
                keyEvents.push({ time: startTime, event: 'RDD Cached', detail: 'labelsRDD cached in memory', type: 'info' });
            } else if (line.startsWith('Total word count:')) {
                keyEvents.push({ time: endTime, event: 'Persistence Complete', detail: line.trim(), type: 'success' });
            }
        }

        // Extract memory warnings
        const warnings = [];
        for (const entry of timeline) {
            if (entry.message.includes('Not enough space to cache') || entry.message.includes('Persisting block')) {
                warnings.push({
                    timestamp: entry.timestamp,
                    message: entry.message
                });
            }
        }

        // Parse RDD lineage from log
        const lineageBlocks = [];
        let capturingLineage = false;
        let currentBlock = { title: '', lines: [] };

        for (const line of logLines) {
            if (line.includes('RDD lineage:') || line.includes('aggregation lineage:')) {
                if (currentBlock.lines.length > 0) lineageBlocks.push({ ...currentBlock });
                currentBlock = { title: line.trim().replace(':', ''), lines: [] };
                capturingLineage = true;
            } else if (capturingLineage) {
                if (line.match(/^\(?\d+\)|^\s+[|+]/) || line.match(/^\s+\|/)) {
                    currentBlock.lines.push(line);
                } else if (line.trim() === '' || !line.match(/^\s/)) {
                    if (currentBlock.lines.length > 0) {
                        lineageBlocks.push({ ...currentBlock });
                        currentBlock = { title: '', lines: [] };
                    }
                    capturingLineage = false;
                }
            }
        }
        if (currentBlock.lines.length > 0) lineageBlocks.push(currentBlock);

        // Parse lineage nodes
        const lineageNodes = [];
        for (const block of lineageBlocks) {
            const nodes = [];
            for (const line of block.lines) {
                const rddMatch = line.match(/(\w+RDD)\[(\d+)\]\s+at\s+(\w+)\s+at\s+(.+?)(?:\s+\[\])?$/);
                if (rddMatch) {
                    nodes.push({
                        type: rddMatch[1],
                        id: parseInt(rddMatch[2]),
                        operation: rddMatch[3],
                        source: rddMatch[4].trim(),
                        indent: line.search(/\S/)
                    });
                }
            }
            if (nodes.length > 0) {
                lineageNodes.push({ title: block.title, nodes });
            }
        }

        // Partition data (from the report)
        const partitions = [
            { id: 1, records: 298935 },
            { id: 2, records: 302167 },
            { id: 3, records: 302477 },
            { id: 4, records: 302238 },
            { id: 5, records: 302745 },
            { id: 6, records: 304769 },
            { id: 7, records: 305138 },
            { id: 8, records: 307818 },
            { id: 9, records: 307444 },
            { id: 10, records: 308217 },
            { id: 11, records: 308548 },
            { id: 12, records: 249504 }
        ];

        this.sparkExecution = {
            info: {
                appName: 'AmazonReviewRDD',
                sparkVersion: '3.5.1',
                scalaVersion: '2.12.18',
                javaVersion: '17.0.20.1',
                executionMode: 'local[4]',
                os: 'Linux 7.0.0-38-generic amd64',
                hostname: 'mithun-HP-EliteBook-840-G6',
                startTime: startTime,
                endTime: endTime,
                memoryCapacity: '2.2 GiB',
                sparkUI: 'http://localhost:4041'
            },
            keyEvents,
            timeline: timeline.slice(0, 50),
            partitions,
            caching: [
                {
                    rdd: 'labelsRDD',
                    method: 'cache()',
                    storageLevel: 'StorageLevel(memory, deserialized, 1 replicas)',
                    count: 3600000,
                    status: 'cached'
                },
                {
                    rdd: 'wordsRDD',
                    method: 'persist(MEMORY_AND_DISK)',
                    storageLevel: 'StorageLevel(disk, memory, deserialized, 1 replicas)',
                    count: 275519555,
                    cachedPartitions: 12,
                    fractionCached: '100%',
                    sizeOnDisk: '1642.4 MiB',
                    sizeInMemory: '0.0 B',
                    status: 'persisted_to_disk'
                }
            ],
            lineage: lineageNodes,
            warnings,
            rawLog,
            summary
        };
    }

    // ── Source Code ──────────────────────────────────────────
    loadSourceCode() {
        const files = {
            scala_rdd: {
                name: 'AmazonReviewRDD.scala',
                path: path.join(PROJECT_ROOT, 'scala', 'AmazonReviewRDD.scala'),
                language: 'scala'
            },
            python_ml: {
                name: 'amazon_review_ml.py',
                path: path.join(PROJECT_ROOT, 'python', 'amazon_review_ml.py'),
                language: 'python'
            },
            python_charts: {
                name: 'generate_charts.py',
                path: path.join(PROJECT_ROOT, 'python', 'generate_charts.py'),
                language: 'python'
            }
        };

        this.sourceCode = {};
        for (const [key, file] of Object.entries(files)) {
            try {
                const content = fs.readFileSync(file.path, 'utf-8');
                const lines = content.split('\n').length;
                const size = fs.statSync(file.path).size;
                this.sourceCode[key] = {
                    name: file.name,
                    language: file.language,
                    content,
                    lines,
                    size
                };
            } catch (err) {
                console.warn(`  ⚠ Could not load ${file.name}: ${err.message}`);
            }
        }
    }

    // ── Paginated Predictions Query ─────────────────────────
    queryPredictions({ page = 1, limit = 50, label, prediction, minProb, maxProb, sort = 'probability', order = 'desc' }) {
        let filtered = this.predictions;

        if (label !== undefined && label !== '' && label !== 'all') {
            const labelVal = parseFloat(label);
            filtered = filtered.filter(p => p.label === labelVal);
        }
        if (prediction !== undefined && prediction !== '' && prediction !== 'all') {
            const predVal = parseFloat(prediction);
            filtered = filtered.filter(p => p.prediction === predVal);
        }
        if (minProb !== undefined && minProb !== '') {
            const min = parseFloat(minProb);
            filtered = filtered.filter(p => p.probability >= min);
        }
        if (maxProb !== undefined && maxProb !== '') {
            const max = parseFloat(maxProb);
            filtered = filtered.filter(p => p.probability <= max);
        }

        // Sort
        const sortKey = sort === 'label' ? 'label' : sort === 'prediction' ? 'prediction' : 'probability';
        const sortDir = order === 'asc' ? 1 : -1;
        filtered.sort((a, b) => (a[sortKey] - b[sortKey]) * sortDir);

        const total = filtered.length;
        const pages = Math.ceil(total / limit);
        const offset = (page - 1) * limit;
        const data = filtered.slice(offset, offset + limit).map((p, i) => ({
            index: offset + i + 1,
            label: p.label,
            prediction: p.prediction,
            probability: +p.probability.toFixed(6),
            correct: p.label === p.prediction
        }));

        return { data, total, page, limit, pages };
    }
}

module.exports = DataLoader;
