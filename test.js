
        // ============================================================
        //  Flow Logic
        // ============================================================
        let activeSkillWeights = {};

        function goToStep(currentId, nextId) {
            const current = document.getElementById(currentId);
            const next = document.getElementById(nextId);
            
            current.classList.remove('slide-up-in');
            current.classList.add('slide-up-out');
            
            setTimeout(() => {
                current.classList.remove('active', 'slide-up-out');
                next.classList.add('active', 'slide-up-in');
            }, 800);
        }

        function processSkills() {
            const text = document.getElementById('job_description').value;
            if (!text.trim()) {
                shakeElement(document.getElementById('job_description'));
                return;
            }
            
            // Extract capitalized words or specific tech terms
            let words = text.match(/\b([A-Z][a-z]+|[A-Z]{2,}|[a-z]+\.js|C\+\+)\b/g) || [];
            words = [...new Set(words)];
            
            const stops = ["The","This","And","For","With","Are","Your","You","Will","Must","Have","Team","Work","Job","Role","In","Of","To","A","An"];
            words = words.filter(w => !stops.includes(w)).slice(0, 10);

            if (words.length === 0) {
                words = ["Communication", "Problem Solving", "Teamwork"];
            }

            const listEl = document.getElementById('skills-weight-list');
            listEl.innerHTML = '';
            
            words.forEach(word => {
                const item = document.createElement('div');
                item.className = 'fs-skill-item';
                item.innerHTML = `
                    <span class="fs-skill-name">${word}</span>
                    <div class="fs-skill-slider-container">
                        <input type="range" class="fs-skill-slider" min="1" max="10" value="5" oninput="this.parentElement.nextElementSibling.textContent=this.value">
                    </div>
                    <span class="fs-skill-value">5</span>
                `;
                listEl.appendChild(item);
            });

            goToStep('step-skills', 'step-weights');
        }

        async function startAnalysisFlow() {
            if (accumulatedFiles.length === 0) {
                shakeElement(document.getElementById('file-drop-zone'));
                return;
            }

            // Gather skill weights
            activeSkillWeights = {};
            const items = document.querySelectorAll('.fs-skill-item');
            items.forEach(item => {
                const name = item.querySelector('.fs-skill-name').textContent;
                const weight = item.querySelector('.fs-skill-slider').value;
                activeSkillWeights[name] = parseInt(weight, 10);
            });

            goToStep('step-upload', 'step-loading');
            
            setTimeout(() => {
                startChunkedUpload();
            }, 800);
        }

        // ============================================================
        //  Ranking Preference Toggle
        // ============================================================
        function toggleRanking() {
            const toggle = document.getElementById('ranking-toggle');
            const input = document.getElementById('ranking_preference');
            const optSkill = document.getElementById('opt-skillset');
            const optExp = document.getElementById('opt-experience');
            
            if (toggle.dataset.state === 'skillset') {
                toggle.dataset.state = 'experience';
                input.value = 'experience';
                optSkill.classList.remove('active');
                optExp.classList.add('active');
            } else {
                toggle.dataset.state = 'skillset';
                input.value = 'skillset';
                optExp.classList.remove('active');
                optSkill.classList.add('active');
            }
        }

        // ============================================================
        //  File Drop Zone Interaction
        // ============================================================
        const dropZone = document.getElementById('file-drop-zone');
        const fileInput = document.getElementById('resume_files');
        const fileHint = document.getElementById('file-count-hint');
        let accumulatedFiles = [];

        dropZone.addEventListener('click', () => fileInput.click());
        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('drag-over');
        });
        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('drag-over');
        });
        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('drag-over');
            addFiles(e.dataTransfer.files);
        });
        fileInput.addEventListener('change', () => {
            addFiles(fileInput.files);
        });

        function addFiles(newFiles) {
            for (const f of newFiles) {
                if (!accumulatedFiles.some(existing => existing.name === f.name && existing.size === f.size)) {
                    accumulatedFiles.push(f);
                }
            }
            updateFileHint();
        }

        function updateFileHint() {
            const count = accumulatedFiles.length;
            if (count > 0) {
                const totalSize = accumulatedFiles.reduce((sum, f) => sum + f.size, 0);
                const sizeStr = formatBytes(totalSize);
                fileHint.textContent = `${count} file${count !== 1 ? 's' : ''} selected (${sizeStr})`;
                dropZone.classList.add('has-files');
            } else {
                fileHint.textContent = 'PDF, DOCX, TXT, or ZIP';
                dropZone.classList.remove('has-files');
            }
        }

        function formatBytes(bytes) {
            if (bytes === 0) return '0 B';
            const k = 1024;
            const sizes = ['B', 'KB', 'MB', 'GB'];
            const i = Math.floor(Math.log(bytes) / Math.log(k));
            return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
        }

        // ============================================================
        //  Chunked Upload Engine
        // ============================================================
        const CHUNK_SIZE = 2 * 1024 * 1024;

        async function startChunkedUpload() {
            const jobDescription = document.getElementById('job_description').value.trim();
            const rankingPreference = document.getElementById('ranking_preference').value;

            try {
                updateProgress('Initializing upload...', 'Creating session', 0, 1);
                const initRes = await fetch('/api/upload/init', { method: 'POST' });
                const { upload_id } = await initRes.json();

                const files = accumulatedFiles;
                let totalChunks = 0;
                let uploadedChunks = 0;

                for (const file of files) {
                    totalChunks += Math.ceil(file.size / CHUNK_SIZE) || 1;
                }

                updateProgress('Uploading files...', `0 / ${totalChunks} chunks`, 0, totalChunks);
                renderFileList(files, []);

                const completedFiles = [];

                for (let fi = 0; fi < files.length; fi++) {
                    const file = files[fi];
                    const fileChunks = Math.ceil(file.size / CHUNK_SIZE) || 1;

                    for (let ci = 0; ci < fileChunks; ci++) {
                        const start = ci * CHUNK_SIZE;
                        const end = Math.min(start + CHUNK_SIZE, file.size);
                        const blob = file.slice(start, end);

                        const formData = new FormData();
                        formData.append('upload_id', upload_id);
                        formData.append('filename', file.name);
                        formData.append('chunk_index', ci);
                        formData.append('total_chunks', fileChunks);
                        formData.append('chunk', blob, file.name);

                        const res = await fetch('/api/upload/chunk', { method: 'POST', body: formData });
                        if (!res.ok) throw new Error(`Upload failed for ${file.name} chunk ${ci}`);

                        const chunkResult = await res.json();
                        uploadedChunks++;

                        const pct = (uploadedChunks / totalChunks) * 100;
                        updateProgress('Uploading files...', `${uploadedChunks} / ${totalChunks} chunks`, uploadedChunks, totalChunks);
                        setProgressBar(pct);

                        if (chunkResult.file_complete) {
                            completedFiles.push(file.name);
                            renderFileList(files, completedFiles);
                        }
                    }
                }

                updateProgress('Analyzing resumes...', 'Reassembling files & AI analysis', 100, 100);
                setProgressBar(100);
                setSpinnerMode('analyzing');

                const finalRes = await fetch('/api/upload/finalize', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ 
                        upload_id, 
                        job_description: jobDescription,
                        ranking_preference: rankingPreference,
                        skill_weights: activeSkillWeights
                    })
                });

                const { job_id, error } = await finalRes.json();
                if (error) throw new Error(error);

                await pollJobStatus(job_id, files.length);

            } catch (err) {
                console.error('Upload error:', err);
                updateProgress('Upload failed', err.message, 0, 0);
                setSpinnerMode('error');
            }
        }

        async function pollJobStatus(jobId, totalFiles) {
            updateProgress('Processing resumes...', 'Parsing documents & computing similarity...', 0, totalFiles);

            const poll = async () => {
                try {
                    const res = await fetch(`/api/job/${jobId}/status`);
                    const data = await res.json();

                    if (data.status === 'completed') {
                        const completeRes = await fetch(`/api/job/${jobId}/complete`, { method: 'POST' });
                        const completeData = await completeRes.json();

                        setSpinnerMode('done');
                        updateProgress('Analysis complete!', `${data.results.length} resumes ranked`, data.total, data.total);
                        setProgressBar(100);

                        setTimeout(() => {
                            window.location.href = completeData.redirect || '/results';
                        }, 800);
                        return;
                    }

                    if (data.total > 0) {
                        const pct = (data.processed / data.total) * 100;
                        updateProgress('Processing resumes...', data.note || `${data.processed} / ${data.total} files parsed`, data.processed, data.total);
                        setProgressBar(pct);
                    }
                    setTimeout(poll, 800);
                } catch (e) {
                    setTimeout(poll, 1500);
                }
            };
            poll();
        }

        function updateProgress(title, subtitle, current, total) {
            document.getElementById('progress-title').textContent = title;
            document.getElementById('progress-subtitle').textContent = subtitle;
            document.getElementById('progress-current').textContent = current;
            document.getElementById('progress-total').textContent = total;
        }

        function setProgressBar(pct) {
            document.getElementById('upload-progress-fill').style.width = Math.min(pct, 100) + '%';
        }

        function setSpinnerMode(mode) {
            const spinner = document.getElementById('progress-spinner');
            spinner.className = 'progress-spinner';
            if (mode) spinner.classList.add('mode-' + mode);
        }

        function renderFileList(allFiles, completedFiles) {
            const container = document.getElementById('progress-file-list');
            container.innerHTML = allFiles.map(f => {
                const done = completedFiles.includes(f.name);
                return `
                    <div class="progress-file-item ${done ? 'completed' : ''}" style="display:flex; justify-content:space-between; padding:8px; border-bottom:1px solid rgba(255,255,255,0.05); color: var(--text-secondary); font-size: 0.85rem;">
                        <span style="display:flex; gap: 8px;">
                            <span class="file-icon" style="color: ${done ? 'var(--success)' : 'inherit'}">${done ? '✓' : '⋯'}</span>
                            <span class="file-name">${f.name}</span>
                        </span>
                        <span class="file-size">${formatBytes(f.size)}</span>
                    </div>
                `;
            }).join('');
        }

        function shakeElement(el) {
            el.classList.add('shake');
            setTimeout(() => el.classList.remove('shake'), 600);
        }

        // ============================================================
        //  Canvas Background Animation (Infinite Frosted Circles)
        // ============================================================
        const canvas = document.getElementById('bg-canvas');
        const ctx = canvas.getContext('2d');
        let particles = [];

        function resizeCanvas() {
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
        }
        window.addEventListener('resize', resizeCanvas);
        resizeCanvas();

        class Particle {
            constructor() {
                this.reset(true);
            }

            reset(initial = false) {
                this.x = Math.random() * canvas.width;
                this.y = initial ? Math.random() * canvas.height : canvas.height + 150;
                this.radius = Math.random() * 100 + 50;
                this.speed = Math.random() * 1.5 + 0.5;
                this.life = Math.random() * 0.4 + 0.1;
                this.color = Math.random() > 0.5 ? '#8b5cf6' : '#06b6d4'; // Purple or Cyan
            }

            update() {
                this.y -= this.speed;
                if (this.y < -this.radius * 2) {
                    this.reset();
                }
            }

            draw() {
                ctx.beginPath();
                ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
                ctx.fillStyle = this.color;
                ctx.globalAlpha = this.life;
                ctx.filter = 'blur(60px)';
                ctx.fill();
            }
        }

        // Initialize particles
        for (let i = 0; i < 12; i++) {
            particles.push(new Particle());
        }

        function animate() {
            // Draw a semi-transparent dark background to create fluid trails
            ctx.globalAlpha = 0.3;
            ctx.fillStyle = '#0a0a0f';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            
            ctx.globalCompositeOperation = 'screen';
            particles.forEach(p => {
                p.update();
                p.draw();
            });
            ctx.globalCompositeOperation = 'source-over';
            
            requestAnimationFrame(animate);
        }
        animate();
    