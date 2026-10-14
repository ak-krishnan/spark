const express = require('express');
const path = require('path');
const DataLoader = require('./data-loader');

const app = express();
const PORT = process.env.PORT || 3000;

// ── Load all real project data on startup ────────────────
const data = new DataLoader();
data.loadAll();

// ── Static files ─────────────────────────────────────────
app.use(express.static(path.join(__dirname, 'public')));

// Serve real chart images from results directory
app.use('/charts', express.static(path.join(__dirname, '..', 'results')));

// ── API Routes ───────────────────────────────────────────

// Health check
app.get('/api/health', (req, res) => {
    res.json({
        status: 'ok',
        uptime: process.uptime(),
        dataLoaded: !!data.metrics,
        predictionsLoaded: data.predictions.length,
        loadTimestamp: data.loadTimestamp
    });
});

// Overview — KPIs, dataset info, project summary
app.get('/api/overview', (req, res) => {
    res.json({
        metrics: data.metrics,
        project: data.projectInfo,
        predictionCount: data.predictions.length,
        dataSource: 'Real Spark execution output (2026-10-13)',
        loadTimestamp: data.loadTimestamp
    });
});

// Model metrics
app.get('/api/metrics', (req, res) => {
    res.json({
        metrics: data.metrics,
        model: data.projectInfo.model,
        features: data.projectInfo.features
    });
});

// Prediction statistics (confusion matrix, ROC, histograms)
app.get('/api/predictions/stats', (req, res) => {
    res.json(data.predictionStats);
});

// Paginated prediction records
app.get('/api/predictions', (req, res) => {
    const { page = 1, limit = 50, label, prediction, minProb, maxProb, sort, order } = req.query;
    const result = data.queryPredictions({
        page: parseInt(page),
        limit: Math.min(parseInt(limit) || 50, 200),
        label,
        prediction,
        minProb,
        maxProb,
        sort: sort || 'probability',
        order: order || 'desc'
    });
    res.json(result);
});

// Spark execution data
app.get('/api/spark/execution', (req, res) => {
    const exec = data.sparkExecution;
    res.json({
        info: exec.info,
        keyEvents: exec.keyEvents,
        partitions: exec.partitions,
        caching: exec.caching,
        lineage: exec.lineage,
        warningCount: exec.warnings.length,
        warnings: exec.warnings.slice(0, 50),
        summary: exec.summary
    });
});

// Spark raw log
app.get('/api/spark/log', (req, res) => {
    res.json({
        log: data.sparkExecution.rawLog,
        lines: data.sparkExecution.rawLog.split('\n').length
    });
});

// Source code
app.get('/api/source', (req, res) => {
    res.json(data.sourceCode);
});

// Catch-all: serve frontend
app.get('*', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// ── Start server ─────────────────────────────────────────
app.listen(PORT, () => {
    console.log(`\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`);
    console.log(`  Amazon Spark Dashboard`);
    console.log(`  http://localhost:${PORT}`);
    console.log(`  Data: ${data.predictions.length.toLocaleString()} real predictions loaded`);
    console.log(`━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n`);
});
