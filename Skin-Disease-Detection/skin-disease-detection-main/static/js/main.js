/**
 * DermoAI - Skin Disease Neural Diagnostics
 * Main Application Logic & Interactivity Controller
 */

document.addEventListener('DOMContentLoaded', () => {
    // App State Store
    const state = {
        selectedModel: 'Ensemble',
        currentFile: null,
        currentBase64: null,
        currentSampleUrl: null,
        currentResult: null,
        sessionHistory: [],
        webcamStream: null
    };

    // DOM Elements
    const elements = {
        navBtns: document.querySelectorAll('.nav-btn'),
        tabPanes: document.querySelectorAll('.tab-pane'),
        modelSelect: document.getElementById('modelSelect'),
        
        // Input Elements
        dropzone: document.getElementById('dropzone'),
        fileInput: document.getElementById('fileInput'),
        webcamBtn: document.getElementById('webcamBtn'),
        samplesContainer: document.getElementById('samplesContainer'),
        
        // Preview Canvas
        previewWrapper: document.getElementById('previewWrapper'),
        imagePreview: document.getElementById('imagePreview'),
        heatmapCanvas: document.getElementById('heatmapCanvas'),
        heatmapToggle: document.getElementById('heatmapToggle'),
        clearImgBtn: document.getElementById('clearImgBtn'),
        analyzeBtn: document.getElementById('analyzeBtn'),
        
        // Results Elements
        emptyState: document.getElementById('emptyState'),
        loadingState: document.getElementById('loadingState'),
        loadingMsg: document.getElementById('loadingMsg'),
        resultsContent: document.getElementById('resultsContent'),
        printReportBtn: document.getElementById('printReportBtn'),
        
        // Results Card Fields
        resDiseaseName: document.getElementById('resDiseaseName'),
        resFullDiseaseName: document.getElementById('resFullDiseaseName'),
        resRiskBadge: document.getElementById('resRiskBadge'),
        resRiskLevel: document.getElementById('resRiskLevel'),
        resConfidenceVal: document.getElementById('resConfidenceVal'),
        resConfidenceBar: document.getElementById('resConfidenceBar'),
        resRiskAlert: document.getElementById('resRiskAlert'),
        resRiskDesc: document.getElementById('resRiskDesc'),
        breakdownList: document.getElementById('breakdownList'),
        
        // Detail Accordion Tabs
        detailTabBtns: document.querySelectorAll('.detail-tab-btn'),
        detailPanes: document.querySelectorAll('.detail-pane'),
        resSymptomsList: document.getElementById('resSymptomsList'),
        resRecommendationsList: document.getElementById('resRecommendationsList'),
        resLocationsTags: document.getElementById('resLocationsTags'),
        
        // Library & Benchmark Grids
        diseaseLibraryGrid: document.getElementById('diseaseLibraryGrid'),
        metricsGrid: document.getElementById('metricsGrid'),
        metricsTableBody: document.getElementById('metricsTableBody'),
        historyList: document.getElementById('historyList'),
        
        // Webcam Modal
        webcamModal: document.getElementById('webcamModal'),
        webcamVideo: document.getElementById('webcamVideo'),
        closeWebcamBtn: document.getElementById('closeWebcamBtn'),
        captureSnapBtn: document.getElementById('captureSnapBtn')
    };

    // Initialize App
    init();

    function init() {
        setupEventListeners();
        loadSampleImages();
        loadDiseaseLibrary();
        loadModelBenchmarks();
    }

    // Event Listener Binding
    function setupEventListeners() {
        // Tab Navigation
        elements.navBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                const targetTab = btn.getAttribute('data-tab');
                elements.navBtns.forEach(b => b.classList.remove('active'));
                elements.tabPanes.forEach(p => p.classList.remove('active'));
                
                btn.classList.add('active');
                document.getElementById(targetTab).classList.add('active');
            });
        });

        // Model Selection Dropdown
        elements.modelSelect.addEventListener('change', (e) => {
            state.selectedModel = e.target.value;
            if (state.currentResult) {
                runPrediction();
            }
        });

        // File Input & Drag and Drop
        elements.fileInput.addEventListener('change', handleFileSelect);
        
        elements.dropzone.addEventListener('dragover', (e) => {
            e.preventDefault();
            elements.dropzone.classList.add('dragover');
        });

        elements.dropzone.addEventListener('dragleave', () => {
            elements.dropzone.classList.remove('dragover');
        });

        elements.dropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            elements.dropzone.classList.remove('dragover');
            if (e.dataTransfer.files && e.dataTransfer.files[0]) {
                handleFile(e.dataTransfer.files[0]);
            }
        });

        // Clear Image
        elements.clearImgBtn.addEventListener('click', resetWorkspace);

        // Analyze Button
        elements.analyzeBtn.addEventListener('click', runPrediction);

        // Heatmap Overlay Toggle
        elements.heatmapToggle.addEventListener('change', (e) => {
            if (e.target.checked) {
                elements.heatmapCanvas.classList.add('active');
            } else {
                elements.heatmapCanvas.classList.remove('active');
            }
        });

        // Clinical Details Accordion Tabs
        elements.detailTabBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                const target = btn.getAttribute('data-target');
                elements.detailTabBtns.forEach(b => b.classList.remove('active'));
                elements.detailPanes.forEach(p => p.classList.add('hidden'));
                elements.detailPanes.forEach(p => p.classList.remove('active'));
                
                btn.classList.add('active');
                const pane = document.getElementById(target);
                pane.classList.remove('hidden');
                pane.classList.add('active');
            });
        });

        // Webcam Triggers
        elements.webcamBtn.addEventListener('click', startWebcam);
        elements.closeWebcamBtn.addEventListener('click', stopWebcam);
        elements.captureSnapBtn.addEventListener('click', captureSnapshot);

        // Print Report
        elements.printReportBtn.addEventListener('click', exportReport);
    }

    // Load Pre-loaded Demo Test Samples
    async function loadSampleImages() {
        try {
            const res = await fetch('/api/sample-images');
            const data = await res.json();
            
            if (data.success && data.samples.length > 0) {
                elements.samplesContainer.innerHTML = '';
                data.samples.forEach(sample => {
                    const chip = document.createElement('div');
                    chip.className = 'sample-chip';
                    chip.innerHTML = `
                        <img src="${sample.url}" alt="${sample.name}" class="sample-thumb">
                        <div class="sample-info">
                            <span class="sample-name">${sample.name}</span>
                            <span class="sample-tag" style="color:${sample.risk_color};">${sample.risk_level}</span>
                        </div>
                    `;
                    chip.addEventListener('click', () => selectSample(sample));
                    elements.samplesContainer.appendChild(chip);
                });
            }
        } catch (err) {
            console.error('Failed to load sample images:', err);
        }
    }

    // Handle Sample Image Selection
    function selectSample(sample) {
        state.currentFile = null;
        state.currentBase64 = null;
        state.currentSampleUrl = sample.url;
        
        elements.imagePreview.src = sample.url;
        elements.previewWrapper.classList.remove('hidden');
        
        // Auto run prediction on sample select
        runPrediction();
    }

    // Handle Local File Selection
    function handleFileSelect(e) {
        if (e.target.files && e.target.files[0]) {
            handleFile(e.target.files[0]);
        }
    }

    function handleFile(file) {
        state.currentFile = file;
        state.currentSampleUrl = null;

        const reader = new FileReader();
        reader.onload = (e) => {
            state.currentBase64 = e.target.result;
            elements.imagePreview.src = e.target.result;
            elements.previewWrapper.classList.remove('hidden');
            
            // Auto run prediction
            runPrediction();
        };
        reader.readAsDataURL(file);
    }

    // Reset Workspace to Initial State
    function resetWorkspace() {
        state.currentFile = null;
        state.currentBase64 = null;
        state.currentSampleUrl = null;
        state.currentResult = null;

        elements.fileInput.value = '';
        elements.imagePreview.src = '';
        elements.previewWrapper.classList.add('hidden');
        elements.resultsContent.classList.add('hidden');
        elements.loadingState.classList.add('hidden');
        elements.emptyState.classList.remove('hidden');
        elements.printReportBtn.classList.add('hidden');
    }

    // Primary AI Inference Function
    async function runPrediction() {
        if (!state.currentFile && !state.currentBase64 && !state.currentSampleUrl) return;

        // UI Loading State
        elements.emptyState.classList.add('hidden');
        elements.resultsContent.classList.add('hidden');
        elements.loadingState.classList.remove('hidden');
        elements.loadingMsg.textContent = `Running ${state.selectedModel} neural network analysis...`;

        try {
            let bodyData;
            let headers = {};

            if (state.currentFile) {
                bodyData = new FormData();
                bodyData.append('file', state.currentFile);
                bodyData.append('model', state.selectedModel);
            } else {
                headers['Content-Type'] = 'application/json';
                bodyData = JSON.stringify({
                    model: state.selectedModel,
                    image_base64: state.currentBase64,
                    image_url: state.currentSampleUrl
                });
            }

            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: headers,
                body: bodyData
            });

            const result = await response.json();

            if (result.success) {
                state.currentResult = result;
                displayResults(result);
                drawHeatmapOverlay(result.heatmap_data);
                saveToHistory(result);
            } else {
                alert('Analysis failed: ' + result.error);
            }
        } catch (err) {
            console.error('Prediction API error:', err);
            alert('Server connection error. Please try again.');
        } finally {
            elements.loadingState.classList.add('hidden');
        }
    }

    // Render Diagnostic Results UI
    function displayResults(data) {
        const top = data.top_prediction;

        elements.resDiseaseName.textContent = top.name;
        elements.resFullDiseaseName.textContent = top.full_name;
        elements.resConfidenceVal.textContent = `${top.percentage}%`;
        elements.resConfidenceBar.style.width = `${top.percentage}%`;

        // Risk Level Badge Styling
        elements.resRiskLevel.textContent = top.risk_level;
        elements.resRiskBadge.className = 'risk-badge';
        if (top.code === 'mel' || top.code === 'bcc') {
            elements.resRiskBadge.classList.add('high-risk');
            elements.resRiskAlert.className = 'risk-alert-box alert-danger';
        } else if (top.code === 'akiec' || top.code === 'vasc') {
            elements.resRiskBadge.classList.add('mod-risk');
            elements.resRiskAlert.className = 'risk-alert-box alert-warning';
        } else {
            elements.resRiskBadge.classList.add('low-risk');
            elements.resRiskAlert.className = 'risk-alert-box alert-success';
        }

        elements.resRiskDesc.textContent = top.risk_description;

        // 7-Class Probability Distribution List
        elements.breakdownList.innerHTML = '';
        data.breakdown.forEach(item => {
            const row = document.createElement('div');
            row.className = 'breakdown-item';
            row.innerHTML = `
                <div class="breakdown-info">
                    <span class="breakdown-name">${item.name} <small>(${item.code.toUpperCase()})</small></span>
                    <span class="breakdown-pct">${item.percentage}%</span>
                </div>
                <div class="breakdown-bar-bg">
                    <div class="breakdown-bar-fill" style="width: ${item.percentage}%; background-color: ${item.risk_color};"></div>
                </div>
            `;
            elements.breakdownList.appendChild(row);
        });

        // Symptoms Checklist
        elements.resSymptomsList.innerHTML = '';
        top.symptoms.forEach(sym => {
            const li = document.createElement('li');
            li.innerHTML = `<i class="fa-solid fa-circle-check"></i> ${sym}`;
            elements.resSymptomsList.appendChild(li);
        });

        // Clinical Action Recommendations
        elements.resRecommendationsList.innerHTML = '';
        top.recommendations.forEach(rec => {
            const li = document.createElement('li');
            li.innerHTML = `<i class="fa-solid fa-user-doctor"></i> ${rec}`;
            elements.resRecommendationsList.appendChild(li);
        });

        // Common Locations Cloud Tags
        elements.resLocationsTags.innerHTML = '';
        top.locations.forEach(loc => {
            const tag = document.createElement('span');
            tag.className = 'tag-item';
            tag.textContent = loc;
            elements.resLocationsTags.appendChild(tag);
        });

        elements.resultsContent.classList.remove('hidden');
        elements.printReportBtn.classList.remove('hidden');
    }

    // Render Heatmap Canvas Overlay
    function drawHeatmapOverlay(data) {
        if (!data) return;
        const canvas = elements.heatmapCanvas;
        const ctx = canvas.getContext('2d');
        
        canvas.width = elements.imagePreview.naturalWidth || 300;
        canvas.height = elements.imagePreview.naturalHeight || 300;

        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Draw radial glowing gradient simulating Grad-CAM lesion focus
        const scaleX = canvas.width / 100;
        const scaleY = canvas.height / 100;
        const cx = data.center_x * scaleX;
        const cy = data.center_y * scaleY;
        const r = data.radius * scaleX;

        const grad = ctx.createRadialGradient(cx, cy, 5, cx, cy, r);
        grad.addColorStop(0, 'rgba(239, 68, 68, 0.85)');
        grad.addColorStop(0.5, 'rgba(245, 158, 11, 0.5)');
        grad.addColorStop(1, 'rgba(14, 165, 233, 0)');

        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.arc(cx, cy, r, 0, Math.PI * 2);
        ctx.fill();
    }

    // Load Disease Knowledge Library
    async function loadDiseaseLibrary() {
        try {
            const res = await fetch('/api/diseases');
            const data = await res.json();

            if (data.success) {
                elements.diseaseLibraryGrid.innerHTML = '';
                data.diseases.forEach(dis => {
                    const card = document.createElement('div');
                    card.className = 'disease-card';
                    card.innerHTML = `
                        <div class="disease-card-header">
                            <h3>${dis.name}</h3>
                            <span class="risk-badge" style="background:${dis.risk_color}22; color:${dis.risk_color}; border:1px solid ${dis.risk_color}66;">
                                ${dis.risk_level}
                            </span>
                        </div>
                        <p>${dis.risk_description}</p>
                        <div style="font-size:12px; color:#94A3B8; margin-top:10px;">
                            <strong>Model Metrics:</strong> Precision: ${(dis.precision*100).toFixed(0)}% | Recall: ${(dis.recall*100).toFixed(0)}% | F1: ${dis.f1_score}
                        </div>
                    `;
                    elements.diseaseLibraryGrid.appendChild(card);
                });
            }
        } catch (err) {
            console.error('Failed to load disease library:', err);
        }
    }

    // Load Deep Learning Architecture Benchmarks
    async function loadModelBenchmarks() {
        try {
            const res = await fetch('/api/models');
            const data = await res.json();

            if (data.success) {
                elements.metricsGrid.innerHTML = '';
                Object.keys(data.metrics).forEach(key => {
                    const m = data.metrics[key];
                    const card = document.createElement('div');
                    card.className = 'metric-card';
                    card.innerHTML = `
                        <h4>${m.name}</h4>
                        <div class="metric-score">${(m.accuracy * 100).toFixed(2)}%</div>
                        <p style="font-size:12px; color:#94A3B8;">Validation Loss: ${m.loss}</p>
                        <p style="font-size:13px; margin-top:8px;">${m.description}</p>
                    `;
                    elements.metricsGrid.appendChild(card);
                });
            }
        } catch (err) {
            console.error('Failed to load model metrics:', err);
        }
    }

    // Session History Management
    function saveToHistory(result) {
        const top = result.top_prediction;
        const entry = {
            id: Date.now(),
            time: new Date().toLocaleTimeString(),
            name: top.name,
            confidence: top.percentage,
            risk_level: top.risk_level,
            risk_color: top.risk_color,
            img_src: elements.imagePreview.src
        };

        state.sessionHistory.unshift(entry);
        renderHistory();
    }

    function renderHistory() {
        if (state.sessionHistory.length === 0) return;
        
        elements.historyList.innerHTML = '';
        state.sessionHistory.forEach(item => {
            const card = document.createElement('div');
            card.className = 'history-card glass-panel';
            card.style.padding = '14px';
            card.style.display = 'flex';
            card.style.alignItems = 'center';
            card.style.gap = '14px';

            card.innerHTML = `
                <img src="${item.img_src}" style="width:50px; height:50px; border-radius:8px; object-fit:cover;">
                <div style="flex:1;">
                    <strong style="display:block; font-size:14px;">${item.name}</strong>
                    <small style="color:#94A3B8;">Time: ${item.time} | Confidence: ${item.confidence}%</small>
                </div>
                <span class="risk-badge" style="color:${item.risk_color}; background:${item.risk_color}22;">
                    ${item.risk_level}
                </span>
            `;
            elements.historyList.appendChild(card);
        });
    }

    // Live Webcam Capture Functions
    async function startWebcam() {
        try {
            state.webcamStream = await navigator.mediaDevices.getUserMedia({ video: true });
            elements.webcamVideo.srcObject = state.webcamStream;
            elements.webcamModal.classList.remove('hidden');
        } catch (err) {
            alert('Unable to access webcam: ' + err.message);
        }
    }

    function stopWebcam() {
        if (state.webcamStream) {
            state.webcamStream.getTracks().forEach(track => track.stop());
            state.webcamStream = null;
        }
        elements.webcamModal.classList.add('hidden');
    }

    function captureSnapshot() {
        const video = elements.webcamVideo;
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth || 640;
        canvas.height = video.videoHeight || 480;

        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        const b64 = canvas.toDataURL('image/jpeg');
        state.currentFile = null;
        state.currentSampleUrl = null;
        state.currentBase64 = b64;

        elements.imagePreview.src = b64;
        elements.previewWrapper.classList.remove('hidden');
        
        stopWebcam();
        runPrediction();
    }

    // Export Diagnostic Report
    function exportReport() {
        if (!state.currentResult) return;
        window.print();
    }
});
