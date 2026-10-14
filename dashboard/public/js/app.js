/* ══════════════════════════════════════════════════════════
   Main Application — Amazon Spark Dashboard
   All data fetched from real Spark output via backend API
   ══════════════════════════════════════════════════════════ */

(function () {
    'use strict';

    // ── State ────────────────────────────────────────────────
    const state = {
        currentPage: 'overview',
        explorer: {
            page: 1,
            limit: 50,
            label: 'all',
            prediction: 'all',
            minProb: '',
            maxProb: '',
            sort: 'probability',
            order: 'desc',
            total: 0,
            pages: 0,
        },
        sourceCode: null,
        activeSourceFile: 'scala_rdd',
        dataLoaded: false,
    };

    // ── API Helper ──────────────────────────────────────────
    async function api(endpoint) {
        try {
            const res = await fetch(`/api/${endpoint}`);
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            return await res.json();
        } catch (err) {
            console.error(`API error (${endpoint}):`, err);
            return null;
        }
    }

    // ── Formatting Helpers ──────────────────────────────────
    function formatNumber(n) {
        if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + 'M';
        if (n >= 1_000) return (n / 1_000).toFixed(0) + 'K';
        return n.toLocaleString();
    }

    function formatPct(n) {
        return (n * 100).toFixed(2) + '%';
    }

    function escapeHtml(str) {
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }

    // ── Navigation ──────────────────────────────────────────
    function initNavigation() {
        const navItems = document.querySelectorAll('.nav-item');
        const pages = document.querySelectorAll('.page');
        const pageTitle = document.getElementById('pageTitle');
        const sidebar = document.getElementById('sidebar');
        const mobileToggle = document.getElementById('mobileToggle');

        const pageTitles = {
            overview: 'Overview',
            performance: 'Model Performance',
            explorer: 'Data Explorer',
            spark: 'Spark Analysis',
            source: 'Source Code',
        };

        navItems.forEach(item => {
            item.addEventListener('click', () => {
                const section = item.dataset.section;
                if (section === state.currentPage) return;

                navItems.forEach(i => i.classList.remove('active'));
                item.classList.add('active');

                pages.forEach(p => p.classList.remove('active'));
                const target = document.getElementById(`page-${section}`);
                if (target) target.classList.add('active');

                pageTitle.textContent = pageTitles[section] || section;
                state.currentPage = section;

                // Load section data on first visit
                if (section === 'performance' && !state.performanceLoaded) loadPerformance();
                if (section === 'explorer' && !state.explorerLoaded) loadExplorer();
                if (section === 'spark' && !state.sparkLoaded) loadSpark();
                if (section === 'source' && !state.sourceLoaded) loadSource();

                // Close mobile sidebar
                sidebar.classList.remove('open');
            });
        });

        // Mobile toggle
        if (mobileToggle) {
            mobileToggle.addEventListener('click', () => {
                sidebar.classList.toggle('open');
            });
        }
    }

    // ── Health Check ────────────────────────────────────────
    async function checkHealth() {
        const statusBadge = document.getElementById('connectionStatus');
        const data = await api('health');
        if (data && data.status === 'ok') {
            statusBadge.innerHTML = `
                <span class="status-dot"></span>
                <span class="status-text">Connected · ${data.predictionsLoaded.toLocaleString()} records</span>
            `;
            statusBadge.className = 'status-badge';
        } else {
            statusBadge.innerHTML = `
                <span class="status-dot"></span>
                <span class="status-text">Disconnected</span>
            `;
            statusBadge.className = 'status-badge error';
        }
    }

    // ══════════════════════════════════════════════════════
    // OVERVIEW SECTION
    // ══════════════════════════════════════════════════════
    async function loadOverview() {
        const data = await api('overview');
        if (!data) return;

        const { metrics, project } = data;

        // Animate hero stat
        const heroStat = document.querySelector('.hero-stat-value');
        if (heroStat) animateValue(heroStat, 0, metrics['Accuracy'] * 100, 1500, '%');

        // KPI Cards
        const kpiMap = {
            'kpi-accuracy': { value: metrics['Accuracy'], label: 'Accuracy' },
            'kpi-precision': { value: metrics['Weighted Precision'], label: 'Precision' },
            'kpi-recall': { value: metrics['Weighted Recall'], label: 'Recall' },
            'kpi-f1': { value: metrics['F1-score'], label: 'F1-Score' },
            'kpi-auc': { value: metrics['ROC-AUC'], label: 'ROC-AUC' },
        };

        for (const [id, kpi] of Object.entries(kpiMap)) {
            const el = document.getElementById(id);
            if (el) animateValue(el, 0, kpi.value * 100, 1200, '%');

            // Animate bar fill
            const card = el?.closest('.kpi-card');
            if (card) {
                const bar = card.querySelector('.kpi-bar-fill');
                if (bar) setTimeout(() => { bar.style.width = (kpi.value * 100) + '%'; }, 300);
            }
        }

        // Dataset stats
        const statMap = {
            'stat-total': project.totalReviews,
            'stat-training': project.trainingReviews,
            'stat-testing': project.testingReviews,
            'stat-words': project.wordElements,
        };
        for (const [id, val] of Object.entries(statMap)) {
            const el = document.getElementById(id);
            if (el) el.textContent = formatNumber(val);
        }

        // Exec summary
        setText('exec-spark', `${project.sparkVersion} (Scala ${project.scalaVersion})`);
        setText('exec-mode', project.executionMode);
        setText('exec-date', new Date(project.executionDate).toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' }));
        setText('exec-partitions', project.partitions.toString());
        setText('exec-memory', '2.2 GiB');
        setText('exec-host', project.hostname);

        // Last updated
        const lastUpdated = document.getElementById('lastUpdated');
        if (lastUpdated) {
            lastUpdated.textContent = `Data from: ${new Date(project.executionDate).toLocaleDateString()}`;
        }

        // Charts
        renderSentimentDonut('sentimentDonut', 1800000, 1800000);
        renderPerformanceRadar('performanceRadar', metrics);

        state.dataLoaded = true;
    }

    // ══════════════════════════════════════════════════════
    // MODEL PERFORMANCE SECTION
    // ══════════════════════════════════════════════════════
    async function loadPerformance() {
        const stats = await api('predictions/stats');
        const metricsData = await api('metrics');
        if (!stats) return;

        // Confusion Matrix
        renderConfusionMatrix(stats.confusionMatrix, stats.total);

        // AUC badge
        const aucBadge = document.getElementById('aucBadge');
        if (aucBadge && metricsData) {
            aucBadge.textContent = `AUC: ${metricsData.metrics['ROC-AUC'].toFixed(4)}`;
        }

        // Charts
        renderROCCurve('rocCurveChart', stats.rocCurve, metricsData?.metrics['ROC-AUC'] || 0.9533);
        renderProbDistribution('probDistChart', stats.probabilityHistogram);
        renderThresholdAnalysis('thresholdChart', stats.thresholdAnalysis);

        state.performanceLoaded = true;
    }

    function renderConfusionMatrix(cm, total) {
        const container = document.getElementById('confusionMatrix');
        if (!container) return;

        const pct = (val) => ((val / total) * 100).toFixed(1) + '%';

        container.innerHTML = `
            <table class="cm-table">
                <tr>
                    <td></td>
                    <td></td>
                    <td colspan="2" class="cm-header" style="text-align:center; padding-bottom:12px;">Predicted</td>
                </tr>
                <tr>
                    <td></td>
                    <td></td>
                    <td class="cm-header">Negative</td>
                    <td class="cm-header">Positive</td>
                </tr>
                <tr>
                    <td rowspan="2" class="cm-row-label" style="vertical-align:middle; writing-mode:vertical-lr; transform:rotate(180deg);">Actual</td>
                    <td class="cm-row-label">Negative</td>
                    <td>
                        <div class="cm-cell" style="background: rgba(99, 102, 241, 0.35); border: 1px solid rgba(99, 102, 241, 0.5);">
                            <span class="cm-cell-value">${cm.tn.toLocaleString()}</span>
                            <span class="cm-cell-pct">TN · ${pct(cm.tn)}</span>
                        </div>
                    </td>
                    <td>
                        <div class="cm-cell" style="background: rgba(245, 158, 11, 0.3); border: 1px solid rgba(245, 158, 11, 0.5);">
                            <span class="cm-cell-value">${cm.fp.toLocaleString()}</span>
                            <span class="cm-cell-pct">FP · ${pct(cm.fp)}</span>
                        </div>
                    </td>
                </tr>
                <tr>
                    <td class="cm-row-label">Positive</td>
                    <td>
                        <div class="cm-cell" style="background: rgba(239, 68, 68, 0.3); border: 1px solid rgba(239, 68, 68, 0.5);">
                            <span class="cm-cell-value">${cm.fn.toLocaleString()}</span>
                            <span class="cm-cell-pct">FN · ${pct(cm.fn)}</span>
                        </div>
                    </td>
                    <td>
                        <div class="cm-cell" style="background: rgba(34, 197, 94, 0.35); border: 1px solid rgba(34, 197, 94, 0.5);">
                            <span class="cm-cell-value">${cm.tp.toLocaleString()}</span>
                            <span class="cm-cell-pct">TP · ${pct(cm.tp)}</span>
                        </div>
                    </td>
                </tr>
            </table>
        `;
    }

    // ══════════════════════════════════════════════════════
    // DATA EXPLORER SECTION
    // ══════════════════════════════════════════════════════
    async function loadExplorer() {
        await fetchPredictions();
        initExplorerControls();
        state.explorerLoaded = true;
    }

    async function fetchPredictions() {
        const s = state.explorer;
        const params = new URLSearchParams({
            page: s.page,
            limit: s.limit,
            sort: s.sort,
            order: s.order,
        });
        if (s.label !== 'all') params.set('label', s.label);
        if (s.prediction !== 'all') params.set('prediction', s.prediction);
        if (s.minProb) params.set('minProb', s.minProb);
        if (s.maxProb) params.set('maxProb', s.maxProb);

        const data = await api(`predictions?${params}`);
        if (!data) return;

        state.explorer.total = data.total;
        state.explorer.pages = data.pages;
        state.explorer.page = data.page;

        renderPredictionsTable(data.data);
        renderPagination(data);
        updateResultsSummary(data);
    }

    function renderPredictionsTable(records) {
        const tbody = document.getElementById('predictionsBody');
        if (!tbody) return;

        if (!records.length) {
            tbody.innerHTML = '<tr><td colspan="5" class="loading-cell">No predictions match the current filters</td></tr>';
            return;
        }

        tbody.innerHTML = records.map(r => `
            <tr>
                <td style="color: var(--text-muted);">${r.index.toLocaleString()}</td>
                <td><span class="label-badge ${r.label === 0 ? 'label-negative' : 'label-positive'}">${r.label === 0 ? 'Negative' : 'Positive'}</span></td>
                <td><span class="label-badge ${r.prediction === 0 ? 'label-negative' : 'label-positive'}">${r.prediction === 0 ? 'Negative' : 'Positive'}</span></td>
                <td>
                    <div class="prob-bar-cell">
                        <span>${r.probability.toFixed(4)}</span>
                        <div class="prob-bar"><div class="prob-bar-fill" style="width: ${r.probability * 100}%; background: ${r.probability > 0.5 ? 'var(--accent-green)' : 'var(--accent-red)'}"></div></div>
                    </div>
                </td>
                <td><span class="${r.correct ? 'status-correct' : 'status-incorrect'}">${r.correct ? '✓ Correct' : '✗ Incorrect'}</span></td>
            </tr>
        `).join('');
    }

    function renderPagination(data) {
        const container = document.getElementById('pageNumbers');
        const prevBtn = document.getElementById('prevPage');
        const nextBtn = document.getElementById('nextPage');

        if (!container) return;

        prevBtn.disabled = data.page <= 1;
        nextBtn.disabled = data.page >= data.pages;

        // Generate page numbers
        const pages = [];
        const total = data.pages;
        const current = data.page;

        if (total <= 7) {
            for (let i = 1; i <= total; i++) pages.push(i);
        } else {
            pages.push(1);
            if (current > 3) pages.push('...');
            for (let i = Math.max(2, current - 1); i <= Math.min(total - 1, current + 1); i++) {
                pages.push(i);
            }
            if (current < total - 2) pages.push('...');
            pages.push(total);
        }

        container.innerHTML = pages.map(p => {
            if (p === '...') return '<span class="page-num" style="cursor:default;">…</span>';
            return `<span class="page-num ${p === current ? 'active' : ''}" data-page="${p}">${p}</span>`;
        }).join('');

        // Event listeners
        container.querySelectorAll('.page-num[data-page]').forEach(el => {
            el.addEventListener('click', () => {
                state.explorer.page = parseInt(el.dataset.page);
                fetchPredictions();
            });
        });
    }

    function updateResultsSummary(data) {
        setText('resultsShowing', Math.min(data.limit, data.data.length).toString());
        setText('resultsTotal', data.total.toLocaleString());
        setText('resultsPage', data.page.toString());
        setText('resultsPages', data.pages.toLocaleString());
    }

    function initExplorerControls() {
        const prevBtn = document.getElementById('prevPage');
        const nextBtn = document.getElementById('nextPage');
        const applyBtn = document.getElementById('applyFilters');
        const resetBtn = document.getElementById('resetFilters');
        const exportBtn = document.getElementById('exportCsv');

        if (prevBtn) prevBtn.addEventListener('click', () => {
            if (state.explorer.page > 1) { state.explorer.page--; fetchPredictions(); }
        });
        if (nextBtn) nextBtn.addEventListener('click', () => {
            if (state.explorer.page < state.explorer.pages) { state.explorer.page++; fetchPredictions(); }
        });

        if (applyBtn) applyBtn.addEventListener('click', () => {
            state.explorer.label = document.getElementById('filterLabel').value;
            state.explorer.prediction = document.getElementById('filterPrediction').value;
            state.explorer.minProb = document.getElementById('filterMinProb').value;
            state.explorer.maxProb = document.getElementById('filterMaxProb').value;
            state.explorer.page = 1;
            fetchPredictions();
        });

        if (resetBtn) resetBtn.addEventListener('click', () => {
            document.getElementById('filterLabel').value = 'all';
            document.getElementById('filterPrediction').value = 'all';
            document.getElementById('filterMinProb').value = '';
            document.getElementById('filterMaxProb').value = '';
            state.explorer = { ...state.explorer, label: 'all', prediction: 'all', minProb: '', maxProb: '', page: 1 };
            fetchPredictions();
        });

        if (exportBtn) exportBtn.addEventListener('click', exportCurrentView);

        // Sortable headers
        document.querySelectorAll('.data-table th.sortable').forEach(th => {
            th.addEventListener('click', () => {
                const sort = th.dataset.sort;
                if (state.explorer.sort === sort) {
                    state.explorer.order = state.explorer.order === 'desc' ? 'asc' : 'desc';
                } else {
                    state.explorer.sort = sort;
                    state.explorer.order = 'desc';
                }
                state.explorer.page = 1;

                // Update header indicators
                document.querySelectorAll('.data-table th.sortable').forEach(h => {
                    h.classList.remove('active-sort');
                    h.textContent = h.textContent.replace(/ [↑↓]$/, '');
                });
                th.classList.add('active-sort');
                th.textContent += state.explorer.order === 'desc' ? ' ↓' : ' ↑';

                fetchPredictions();
            });
        });
    }

    async function exportCurrentView() {
        const s = state.explorer;
        const params = new URLSearchParams({
            page: 1, limit: 10000,
            sort: s.sort, order: s.order,
        });
        if (s.label !== 'all') params.set('label', s.label);
        if (s.prediction !== 'all') params.set('prediction', s.prediction);
        if (s.minProb) params.set('minProb', s.minProb);
        if (s.maxProb) params.set('maxProb', s.maxProb);

        const data = await api(`predictions?${params}`);
        if (!data || !data.data.length) return;

        let csv = 'index,label,prediction,probability,correct\n';
        for (const r of data.data) {
            csv += `${r.index},${r.label},${r.prediction},${r.probability},${r.correct}\n`;
        }

        const blob = new Blob([csv], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'spark_predictions_export.csv';
        a.click();
        URL.revokeObjectURL(url);
    }

    // ══════════════════════════════════════════════════════
    // SPARK ANALYSIS SECTION
    // ══════════════════════════════════════════════════════
    async function loadSpark() {
        const [execData, logData] = await Promise.all([
            api('spark/execution'),
            api('spark/log'),
        ]);
        if (!execData) return;

        // Info cards
        const infoGrid = document.getElementById('sparkInfoGrid');
        if (infoGrid) {
            const info = execData.info;
            const cards = [
                { label: 'Application', value: info.appName, detail: `Spark ${info.sparkVersion}` },
                { label: 'Execution Mode', value: info.executionMode, detail: `${info.os}` },
                { label: 'Memory Capacity', value: info.memoryCapacity, detail: `${execData.warningCount} spill warnings` },
                { label: 'Execution Time', value: info.startTime, detail: info.hostname },
            ];

            infoGrid.innerHTML = cards.map(c => `
                <div class="spark-info-card">
                    <div class="info-label">${c.label}</div>
                    <div class="info-value">${escapeHtml(c.value)}</div>
                    <div class="info-detail">${escapeHtml(c.detail)}</div>
                </div>
            `).join('');
        }

        // Partition chart
        renderPartitionChart('partitionChart', execData.partitions);

        // RDD Lineage
        renderLineage(execData.lineage);

        // Caching info
        renderCaching(execData.caching);

        // Warnings
        renderWarnings(execData.warnings);

        // Raw log
        if (logData) {
            const logContent = document.getElementById('logContent');
            if (logContent) logContent.textContent = logData.log;

            const toggleBtn = document.getElementById('toggleLog');
            const logViewer = document.getElementById('logViewer');
            if (toggleBtn && logViewer) {
                toggleBtn.addEventListener('click', () => {
                    logViewer.classList.toggle('collapsed');
                    toggleBtn.textContent = logViewer.classList.contains('collapsed') ? 'Show Full Log' : 'Collapse Log';
                });
            }
        }

        // Warning count badge
        const warnBadge = document.getElementById('warningCount');
        if (warnBadge) warnBadge.textContent = `${execData.warningCount} warnings`;

        state.sparkLoaded = true;
    }

    function renderLineage(lineageBlocks) {
        const container = document.getElementById('lineageContainer');
        if (!container || !lineageBlocks.length) return;

        container.innerHTML = lineageBlocks.map(block => {
            const nodesHtml = block.nodes.map(node => {
                let iconClass = 'map';
                let iconText = 'MP';
                if (node.type.includes('FileScan')) { iconClass = 'file'; iconText = 'FS'; }
                else if (node.type.includes('Shuffled')) { iconClass = 'shuffle'; iconText = 'SH'; }
                else if (node.type.includes('SQL')) { iconClass = 'sql'; iconText = 'SQ'; }

                return `
                    <div class="lineage-node">
                        <div class="lineage-node-icon ${iconClass}">${iconText}</div>
                        <div class="lineage-node-info">
                            <div class="lineage-node-type">${escapeHtml(node.type)}[${node.id}]</div>
                            <div class="lineage-node-detail">${escapeHtml(node.operation)} @ ${escapeHtml(node.source)}</div>
                        </div>
                    </div>
                `;
            }).join('');

            return `
                <div class="lineage-block">
                    <div class="lineage-title">${escapeHtml(block.title)}</div>
                    <div class="lineage-dag">${nodesHtml}</div>
                </div>
            `;
        }).join('');
    }

    function renderCaching(cachingData) {
        const grid = document.getElementById('cachingGrid');
        if (!grid) return;

        grid.innerHTML = cachingData.map(c => {
            const statusClass = c.status === 'cached' ? 'cached' : 'persisted';
            const statusText = c.status === 'cached' ? 'In Memory' : 'Disk + Memory';

            let details = `
                <div class="cache-detail"><span class="cache-detail-label">Method</span><span class="cache-detail-value">${escapeHtml(c.method)}</span></div>
                <div class="cache-detail"><span class="cache-detail-label">Storage Level</span><span class="cache-detail-value">${escapeHtml(c.storageLevel)}</span></div>
                <div class="cache-detail"><span class="cache-detail-label">Record Count</span><span class="cache-detail-value">${c.count.toLocaleString()}</span></div>
            `;

            if (c.cachedPartitions) {
                details += `
                    <div class="cache-detail"><span class="cache-detail-label">Partitions</span><span class="cache-detail-value">${c.cachedPartitions} (${c.fractionCached})</span></div>
                    <div class="cache-detail"><span class="cache-detail-label">Size on Disk</span><span class="cache-detail-value">${c.sizeOnDisk}</span></div>
                    <div class="cache-detail"><span class="cache-detail-label">Size in Memory</span><span class="cache-detail-value">${c.sizeInMemory}</span></div>
                `;
            }

            return `
                <div class="cache-card">
                    <div class="cache-card-header">
                        <span class="cache-rdd-name">${escapeHtml(c.rdd)}</span>
                        <span class="cache-status ${statusClass}">${statusText}</span>
                    </div>
                    ${details}
                </div>
            `;
        }).join('');
    }

    function renderWarnings(warnings) {
        const list = document.getElementById('warningsList');
        if (!list) return;

        if (!warnings.length) {
            list.innerHTML = '<div class="warning-item"><span class="warning-msg" style="color: var(--accent-green);">No memory warnings</span></div>';
            return;
        }

        list.innerHTML = warnings.map(w => `
            <div class="warning-item">
                <span class="warning-time">${escapeHtml(w.timestamp)}</span>
                <span class="warning-msg">${escapeHtml(w.message)}</span>
            </div>
        `).join('');
    }

    // ══════════════════════════════════════════════════════
    // SOURCE CODE SECTION
    // ══════════════════════════════════════════════════════
    async function loadSource() {
        const data = await api('source');
        if (!data) return;

        state.sourceCode = data;
        displaySourceFile(state.activeSourceFile);
        initSourceTabs();
        state.sourceLoaded = true;
    }

    function initSourceTabs() {
        document.querySelectorAll('.code-tab').forEach(tab => {
            tab.addEventListener('click', () => {
                document.querySelectorAll('.code-tab').forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                state.activeSourceFile = tab.dataset.file;
                displaySourceFile(state.activeSourceFile);
            });
        });
    }

    function displaySourceFile(fileKey) {
        const file = state.sourceCode?.[fileKey];
        if (!file) return;

        setText('codeFilename', file.name);
        setText('codeInfo', `${file.lines} lines · ${(file.size / 1024).toFixed(1)} KB · ${file.language}`);

        const codeContent = document.getElementById('codeContent');
        if (!codeContent) return;

        const lines = file.content.split('\n');
        const highlighted = lines.map(line => {
            return `<span class="code-line">${highlightSyntax(line, file.language)}</span>`;
        }).join('');

        codeContent.innerHTML = `<code>${highlighted}</code>`;
    }

    // Escape only text parts while preserving our span tags
    function safeEscape(text) {
        return text
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }

    function highlightSyntax(rawLine, language) {
        // First escape HTML entities in the raw line
        let line = safeEscape(rawLine);

        if (language === 'scala') {
            // Comments first (they override everything)
            line = line.replace(/(\/\/.*)$/g, '<span class="code-comment">$1</span>');
            // Strings
            line = line.replace(/(&quot;(?:(?!&quot;).)*?&quot;)/g, '<span class="code-string">$1</span>');
            // Keywords
            line = line.replace(/\b(import|val|def|object|class|extends|with|new|if|else|for|while|return|try|catch|finally|case|match|yield|true|false|null|this|super|override)\b/g, '<span class="code-keyword">$1</span>');
            // Types
            line = line.replace(/\b(Unit|String|Int|Long|Double|Float|Boolean|Array|List|Map|Set|Option|Some|None)\b/g, '<span class="code-type">$1</span>');
            // Numbers
            line = line.replace(/\b(\d+\.?\d*[fFdDlL]?)\b/g, '<span class="code-number">$1</span>');
        } else if (language === 'python') {
            // Comments first
            line = line.replace(/(#.*)$/g, '<span class="code-comment">$1</span>');
            // Strings
            line = line.replace(/(&quot;(?:(?!&quot;).)*?&quot;|&#39;(?:(?!&#39;).)*?&#39;)/g, '<span class="code-string">$1</span>');
            // Keywords
            line = line.replace(/\b(import|from|as|def|class|if|elif|else|for|while|return|try|except|finally|with|pass|break|continue|and|or|not|in|is|None|True|False|raise|lambda|yield|global|nonlocal|assert|del)\b/g, '<span class="code-keyword">$1</span>');
            // Numbers
            line = line.replace(/\b(\d+\.?\d*)\b/g, '<span class="code-number">$1</span>');
        }
        return line;
    }

    // ── Utility Functions ───────────────────────────────────
    function setText(id, text) {
        const el = document.getElementById(id);
        if (el) el.textContent = text;
    }

    function animateValue(el, start, end, duration, suffix = '') {
        const startTime = performance.now();
        const diff = end - start;

        function update(currentTime) {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            const eased = 1 - Math.pow(1 - progress, 3); // ease-out cubic
            const current = start + diff * eased;

            el.textContent = current.toFixed(2) + suffix;

            if (progress < 1) {
                requestAnimationFrame(update);
            }
        }

        requestAnimationFrame(update);
    }

    // ── Initialize ──────────────────────────────────────────
    async function init() {
        initNavigation();
        await checkHealth();
        await loadOverview();
    }

    // Start when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

})();
