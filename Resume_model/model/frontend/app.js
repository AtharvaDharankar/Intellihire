document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const fileCount = document.getElementById('file-count');
    const form = document.getElementById('job-form');
    const submitBtn = document.getElementById('submit-btn');
    const btnText = submitBtn.querySelector('.btn-text');
    const loader = submitBtn.querySelector('.loader');
    
    const resultsPanel = document.getElementById('results-panel');
    const resultsMeta = document.getElementById('results-meta');
    const candidateList = document.getElementById('candidate-list');

    // Currently selected files
    let selectedFiles = [];

    // Drag & Drop Handlers
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => {
            dropZone.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => {
            dropZone.classList.remove('dragover');
        }, false);
    });

    dropZone.addEventListener('drop', (e) => {
        handleFiles(e.dataTransfer.files);
    });

    fileInput.addEventListener('change', (e) => {
        handleFiles(e.target.files);
    });

    function handleFiles(files) {
        const fileArray = Array.from(files);
        // Validate extensions
        const validExts = ['.pdf', '.docx', '.txt'];
        const validFiles = fileArray.filter(f => {
            const ext = '.' + f.name.split('.').pop().toLowerCase();
            return validExts.includes(ext);
        });

        if (validFiles.length < fileArray.length) {
            alert('Some files were ignored. Only .pdf, .docx, and .txt are supported.');
        }

        // Add to selected files, limit to 100
        selectedFiles = [...selectedFiles, ...validFiles].slice(0, 100);
        
        updateFileCount();
    }

    function updateFileCount() {
        if (selectedFiles.length > 0) {
            fileCount.textContent = `${selectedFiles.length} file${selectedFiles.length === 1 ? '' : 's'} selected`;
            fileCount.classList.remove('hidden');
        } else {
            fileCount.classList.add('hidden');
        }
    }

    // Form Submission
    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        if (selectedFiles.length === 0) {
            alert('Please upload at least one resume.');
            return;
        }

        const formData = new FormData();
        
        // Append all form fields
        formData.append('title', document.getElementById('title').value);
        formData.append('company', document.getElementById('company').value);
        formData.append('description', document.getElementById('description').value);
        formData.append('required_skills', document.getElementById('required_skills').value);
        formData.append('preferred_skills', document.getElementById('preferred_skills').value);
        formData.append('min_years_experience', document.getElementById('min_years_experience').value);
        formData.append('seniority_target', document.getElementById('seniority_target').value);

        // Append all files
        selectedFiles.forEach(file => {
            formData.append('files', file);
        });

        // UI Loading state
        submitBtn.disabled = true;
        btnText.textContent = 'Processing...';
        loader.classList.remove('hidden');
        resultsPanel.classList.add('hidden');

        try {
            // Send to FastAPI backend
            const response = await fetch('http://localhost:8000/rank/files', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.detail || 'Failed to rank resumes.');
            }

            const data = await response.json();
            renderResults(data);
            
        } catch (error) {
            console.error('Error:', error);
            alert(`Error processing resumes: ${error.message}`);
        } finally {
            submitBtn.disabled = false;
            btnText.textContent = 'Analyze & Rank Candidates';
            loader.classList.add('hidden');
        }
    });

    function renderResults(data) {
        resultsMeta.innerHTML = `
            Processed <strong>${data.total_resumes_processed}</strong> resumes 
            in <strong>${data.processing_time_seconds}s</strong> 
            for ${data.job_title}
        `;

        candidateList.innerHTML = '';

        if (!data.ranked_candidates || data.ranked_candidates.length === 0) {
            candidateList.innerHTML = '<p>No candidates were successfully ranked.</p>';
            resultsPanel.classList.remove('hidden');
            return;
        }

        data.ranked_candidates.forEach((candidate, index) => {
            // Determine score class
            let scoreClass = 'score-med';
            if (candidate.fit_score >= 75) scoreClass = 'score-high';
            else if (candidate.fit_score < 40) scoreClass = 'score-low';

            // Generate skills tags
            const skillsHtml = candidate.matched_skills.slice(0, 8).map(s => 
                `<span class="skill-tag">${s}</span>`
            ).join('');

            // File URL link
            const fileUrl = candidate.filename ? `http://localhost:8000/uploads/${candidate.filename}` : '#';

            const card = document.createElement('div');
            card.className = 'candidate-card';
            card.style.animationDelay = `${index * 0.05}s`; // staggered animation
            
            card.innerHTML = `
                <div class="card-top">
                    <div class="rank-badge">#${candidate.rank}</div>
                    
                    <div class="candidate-info">
                        <a href="${fileUrl}" target="_blank" class="candidate-name" title="Open PDF">
                            ${candidate.candidate_name || 'Unknown Candidate'}
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"></path><polyline points="15 3 21 3 21 9"></polyline><line x1="10" y1="14" x2="21" y2="3"></line></svg>
                        </a>
                        <div class="skills-container">
                            ${skillsHtml || '<span style="color:var(--text-muted);font-size:0.8rem;">No required skills detected</span>'}
                        </div>
                    </div>
                    
                    <div class="score-container">
                        <div class="fit-score ${scoreClass}">${Math.round(candidate.fit_score)}</div>
                        <div class="score-label">Fit Score</div>
                    </div>
                </div>
                
                <div class="summary-box">
                    ${candidate.summary}
                </div>
            `;
            
            candidateList.appendChild(card);
        });

        resultsPanel.classList.remove('hidden');
        resultsPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
});
