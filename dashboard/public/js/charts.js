/* ══════════════════════════════════════════════════════════
   Chart Configurations — All data from real Spark outputs
   ══════════════════════════════════════════════════════════ */

const ChartColors = {
    blue: '#6366f1',
    blueAlpha: 'rgba(99, 102, 241, 0.2)',
    cyan: '#22d3ee',
    cyanAlpha: 'rgba(34, 211, 238, 0.2)',
    green: '#22c55e',
    greenAlpha: 'rgba(34, 197, 94, 0.2)',
    amber: '#f59e0b',
    amberAlpha: 'rgba(245, 158, 11, 0.2)',
    red: '#ef4444',
    redAlpha: 'rgba(239, 68, 68, 0.2)',
    purple: '#a78bfa',
    purpleAlpha: 'rgba(167, 139, 250, 0.2)',
    textPrimary: '#e2e8f0',
    textSecondary: '#94a3b8',
    textMuted: '#64748b',
    gridLine: 'rgba(255, 255, 255, 0.05)',
};

Chart.defaults.color = ChartColors.textSecondary;
Chart.defaults.borderColor = ChartColors.gridLine;
Chart.defaults.font.family = "'Inter', sans-serif";

const chartInstances = {};

function destroyChart(id) {
    if (chartInstances[id]) {
        chartInstances[id].destroy();
        delete chartInstances[id];
    }
}

// ── Sentiment Distribution Doughnut ──────────────────────
function renderSentimentDonut(canvasId, positive, negative) {
    destroyChart(canvasId);
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    chartInstances[canvasId] = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Negative Reviews', 'Positive Reviews'],
            datasets: [{
                data: [negative, positive],
                backgroundColor: [
                    ChartColors.red,
                    ChartColors.green,
                ],
                borderColor: 'transparent',
                borderWidth: 0,
                hoverOffset: 8,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '68%',
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        padding: 20,
                        usePointStyle: true,
                        pointStyleWidth: 10,
                        font: { size: 12, weight: 500 }
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(10, 14, 26, 0.95)',
                    titleColor: ChartColors.textPrimary,
                    bodyColor: ChartColors.textSecondary,
                    borderColor: 'rgba(99, 102, 241, 0.2)',
                    borderWidth: 1,
                    cornerRadius: 8,
                    padding: 12,
                    callbacks: {
                        label: ctx => `${ctx.label}: ${ctx.parsed.toLocaleString()} (50%)`
                    }
                }
            }
        }
    });
}

// ── Performance Radar ───────────────────────────────────
function renderPerformanceRadar(canvasId, metrics) {
    destroyChart(canvasId);
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    chartInstances[canvasId] = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
            datasets: [{
                label: 'Logistic Regression',
                data: [
                    metrics['Accuracy'],
                    metrics['Weighted Precision'],
                    metrics['Weighted Recall'],
                    metrics['F1-score'],
                    metrics['ROC-AUC']
                ],
                backgroundColor: ChartColors.blueAlpha,
                borderColor: ChartColors.blue,
                borderWidth: 2,
                pointBackgroundColor: ChartColors.blue,
                pointBorderColor: '#fff',
                pointBorderWidth: 1,
                pointRadius: 5,
                pointHoverRadius: 7,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                r: {
                    beginAtZero: false,
                    min: 0.7,
                    max: 1.0,
                    ticks: {
                        stepSize: 0.05,
                        color: ChartColors.textMuted,
                        font: { size: 10 },
                        backdropColor: 'transparent',
                    },
                    grid: { color: ChartColors.gridLine },
                    angleLines: { color: ChartColors.gridLine },
                    pointLabels: {
                        color: ChartColors.textSecondary,
                        font: { size: 11, weight: 500 }
                    }
                }
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(10, 14, 26, 0.95)',
                    titleColor: ChartColors.textPrimary,
                    bodyColor: ChartColors.textSecondary,
                    borderColor: 'rgba(99, 102, 241, 0.2)',
                    borderWidth: 1,
                    cornerRadius: 8,
                    padding: 12,
                    callbacks: {
                        label: ctx => `${ctx.label}: ${(ctx.parsed.r * 100).toFixed(2)}%`
                    }
                }
            }
        }
    });
}

// ── ROC Curve ───────────────────────────────────────────
function renderROCCurve(canvasId, rocPoints, auc) {
    destroyChart(canvasId);
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    chartInstances[canvasId] = new Chart(ctx, {
        type: 'line',
        data: {
            datasets: [
                {
                    label: `Logistic Regression (AUC = ${auc.toFixed(4)})`,
                    data: rocPoints.map(p => ({ x: p.fpr, y: p.tpr })),
                    borderColor: ChartColors.blue,
                    backgroundColor: ChartColors.blueAlpha,
                    borderWidth: 2.5,
                    fill: true,
                    tension: 0.1,
                    pointRadius: 0,
                    pointHoverRadius: 5,
                },
                {
                    label: 'Random Classifier',
                    data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
                    borderColor: ChartColors.textMuted,
                    borderWidth: 1.5,
                    borderDash: [6, 4],
                    fill: false,
                    pointRadius: 0,
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    type: 'linear',
                    title: { display: true, text: 'False Positive Rate', color: ChartColors.textSecondary, font: { size: 12 } },
                    min: 0, max: 1,
                    ticks: { color: ChartColors.textMuted, font: { size: 10 } },
                    grid: { color: ChartColors.gridLine },
                },
                y: {
                    type: 'linear',
                    title: { display: true, text: 'True Positive Rate', color: ChartColors.textSecondary, font: { size: 12 } },
                    min: 0, max: 1,
                    ticks: { color: ChartColors.textMuted, font: { size: 10 } },
                    grid: { color: ChartColors.gridLine },
                }
            },
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: { padding: 16, usePointStyle: true, font: { size: 11 } }
                },
                tooltip: {
                    backgroundColor: 'rgba(10, 14, 26, 0.95)',
                    borderColor: 'rgba(99, 102, 241, 0.2)',
                    borderWidth: 1,
                    cornerRadius: 8,
                    padding: 12,
                    callbacks: {
                        label: ctx => `FPR: ${ctx.parsed.x.toFixed(3)}, TPR: ${ctx.parsed.y.toFixed(3)}`
                    }
                }
            }
        }
    });
}

// ── Probability Distribution Histogram ──────────────────
function renderProbDistribution(canvasId, histogram) {
    destroyChart(canvasId);
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    const labels = histogram.labels.map((l, i) => {
        if (i % 5 === 0) return parseFloat(l).toFixed(1);
        return '';
    });

    chartInstances[canvasId] = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: histogram.labels,
            datasets: [
                {
                    label: 'Actual Negative (0)',
                    data: histogram.negative,
                    backgroundColor: ChartColors.redAlpha,
                    borderColor: ChartColors.red,
                    borderWidth: 1,
                    borderRadius: 2,
                },
                {
                    label: 'Actual Positive (1)',
                    data: histogram.positive,
                    backgroundColor: ChartColors.greenAlpha,
                    borderColor: ChartColors.green,
                    borderWidth: 1,
                    borderRadius: 2,
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    title: { display: true, text: 'Positive Probability', color: ChartColors.textSecondary, font: { size: 12 } },
                    ticks: {
                        color: ChartColors.textMuted,
                        font: { size: 9 },
                        maxRotation: 0,
                        callback: (value, index) => index % 5 === 0 ? (index / 50).toFixed(1) : ''
                    },
                    grid: { display: false },
                },
                y: {
                    title: { display: true, text: 'Count', color: ChartColors.textSecondary, font: { size: 12 } },
                    ticks: {
                        color: ChartColors.textMuted,
                        font: { size: 10 },
                        callback: v => v >= 1000 ? (v / 1000).toFixed(0) + 'K' : v
                    },
                    grid: { color: ChartColors.gridLine },
                }
            },
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: { padding: 16, usePointStyle: true, font: { size: 11 } }
                },
                tooltip: {
                    backgroundColor: 'rgba(10, 14, 26, 0.95)',
                    borderColor: 'rgba(99, 102, 241, 0.2)',
                    borderWidth: 1,
                    cornerRadius: 8,
                    padding: 12,
                    callbacks: {
                        title: ctx => `Probability: ${ctx[0].label}`,
                        label: ctx => `${ctx.dataset.label}: ${ctx.parsed.y.toLocaleString()}`
                    }
                }
            }
        }
    });
}

// ── Threshold Analysis ──────────────────────────────────
function renderThresholdAnalysis(canvasId, thresholdData) {
    destroyChart(canvasId);
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    chartInstances[canvasId] = new Chart(ctx, {
        type: 'line',
        data: {
            labels: thresholdData.map(t => t.threshold.toFixed(2)),
            datasets: [
                {
                    label: 'Precision',
                    data: thresholdData.map(t => t.precision),
                    borderColor: ChartColors.blue,
                    backgroundColor: ChartColors.blueAlpha,
                    borderWidth: 2,
                    tension: 0.3,
                    pointRadius: 3,
                    pointHoverRadius: 6,
                },
                {
                    label: 'Recall',
                    data: thresholdData.map(t => t.recall),
                    borderColor: ChartColors.green,
                    backgroundColor: ChartColors.greenAlpha,
                    borderWidth: 2,
                    tension: 0.3,
                    pointRadius: 3,
                    pointHoverRadius: 6,
                },
                {
                    label: 'F1-Score',
                    data: thresholdData.map(t => t.f1),
                    borderColor: ChartColors.amber,
                    backgroundColor: ChartColors.amberAlpha,
                    borderWidth: 2,
                    tension: 0.3,
                    pointRadius: 3,
                    pointHoverRadius: 6,
                },
                {
                    label: 'Accuracy',
                    data: thresholdData.map(t => t.accuracy),
                    borderColor: ChartColors.purple,
                    backgroundColor: ChartColors.purpleAlpha,
                    borderWidth: 2,
                    tension: 0.3,
                    pointRadius: 3,
                    pointHoverRadius: 6,
                    borderDash: [4, 4],
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    title: { display: true, text: 'Classification Threshold', color: ChartColors.textSecondary, font: { size: 12 } },
                    ticks: { color: ChartColors.textMuted, font: { size: 10 } },
                    grid: { color: ChartColors.gridLine },
                },
                y: {
                    min: 0, max: 1,
                    title: { display: true, text: 'Score', color: ChartColors.textSecondary, font: { size: 12 } },
                    ticks: { color: ChartColors.textMuted, font: { size: 10 } },
                    grid: { color: ChartColors.gridLine },
                }
            },
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: { padding: 16, usePointStyle: true, font: { size: 11 } }
                },
                tooltip: {
                    backgroundColor: 'rgba(10, 14, 26, 0.95)',
                    borderColor: 'rgba(99, 102, 241, 0.2)',
                    borderWidth: 1,
                    cornerRadius: 8,
                    padding: 12,
                    mode: 'index',
                    intersect: false,
                }
            },
            interaction: { mode: 'index', intersect: false }
        }
    });
}

// ── Partition Distribution ──────────────────────────────
function renderPartitionChart(canvasId, partitions) {
    destroyChart(canvasId);
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    chartInstances[canvasId] = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: partitions.map(p => `P${p.id}`),
            datasets: [{
                label: 'Records per Partition',
                data: partitions.map(p => p.records),
                backgroundColor: partitions.map((p, i) => {
                    const hue = 230 + (i * 10);
                    return `hsla(${hue}, 80%, 65%, 0.6)`;
                }),
                borderColor: partitions.map((p, i) => {
                    const hue = 230 + (i * 10);
                    return `hsla(${hue}, 80%, 65%, 1)`;
                }),
                borderWidth: 1.5,
                borderRadius: 6,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    title: { display: true, text: 'Partition', color: ChartColors.textSecondary, font: { size: 12 } },
                    ticks: { color: ChartColors.textMuted, font: { size: 10 } },
                    grid: { display: false },
                },
                y: {
                    beginAtZero: true,
                    title: { display: true, text: 'Records', color: ChartColors.textSecondary, font: { size: 12 } },
                    ticks: {
                        color: ChartColors.textMuted,
                        font: { size: 10 },
                        callback: v => (v / 1000).toFixed(0) + 'K'
                    },
                    grid: { color: ChartColors.gridLine },
                }
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(10, 14, 26, 0.95)',
                    borderColor: 'rgba(99, 102, 241, 0.2)',
                    borderWidth: 1,
                    cornerRadius: 8,
                    padding: 12,
                    callbacks: {
                        label: ctx => `Records: ${ctx.parsed.y.toLocaleString()}`
                    }
                }
            }
        }
    });
}
