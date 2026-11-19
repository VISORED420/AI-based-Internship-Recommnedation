document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('recommendation-form');
    const liveJobsBtn = document.getElementById('live-jobs-btn');
    const resultsContainer = document.getElementById('results-container');
    const recommendationsDiv = document.getElementById('recommendations');
    const loadingDiv = document.getElementById('loading');
    const errorDiv = document.getElementById('error-message');

    // Carry the same profile over to the live JSearch search page.
    if (liveJobsBtn) {
        liveJobsBtn.addEventListener('click', () => {
            const formData = new FormData(form);
            const params = new URLSearchParams();
            const pairs = [
                ['skills', (formData.get('skills') || '').trim()],
                ['qualification', formData.get('education') || ''],
                ['sector', formData.get('sector_interests') || ''],
                ['location', formData.get('location') || '']
            ];
            pairs.forEach(([key, value]) => { if (value) params.set(key, value); });
            window.location.href = '/jobs' + (params.toString() ? '?' + params.toString() : '');
        });
    }

    form.addEventListener('submit', async (e) => {
        e.preventDefault(); // Stop the form from reloading the page

        // 2. Get form data and map it to the keys the backend expects
        const formData = new FormData(form);
        const candidateData = {
            qualification: formData.get('education'),
            skills: formData.get('skills'),
            sector_interested: formData.get('sector_interests'),
            location_interested: formData.get('location'),
            mode: formData.get('mode') || 'hybrid'
        };

        // Prompt for at least one criterion instead of asking the server for nothing
        if (!candidateData.skills.trim() && !candidateData.sector_interested && !candidateData.location_interested) {
            resultsContainer.classList.remove('hidden');
            loadingDiv.classList.add('hidden');
            recommendationsDiv.innerHTML = '';
            errorDiv.textContent = 'Please enter at least one skill, sector or location to get recommendations.';
            errorDiv.classList.remove('hidden');
            return;
        }

        // 1. Show loading spinner and clear previous results
        resultsContainer.classList.remove('hidden');
        loadingDiv.classList.remove('hidden');
        errorDiv.classList.add('hidden');
        recommendationsDiv.innerHTML = '';

        try {
            // 3. Send the corrected data to the Flask backend
            const response = await fetch('/recommend', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(candidateData),
            });

            // Session expired or not logged in -> send the user to the login form
            if (response.status === 401) {
                window.location.href = '/login?next=' + encodeURIComponent(window.location.pathname);
                return;
            }

            const payload = await response.json().catch(() => null);

            if (!response.ok) {
                // Surface the backend's {"error": "..."} message
                throw new Error((payload && payload.error) || 'Something went wrong with the request.');
            }

            const recommendations = payload || [];

            // 4. Hide loading spinner and display results
            loadingDiv.classList.add('hidden');
            displayRecommendations(recommendations);

        } catch (error) {
            // 5. Handle any errors
            loadingDiv.classList.add('hidden');
            errorDiv.textContent = `Error: ${error.message}`;
            errorDiv.classList.remove('hidden');
        }
    });

    function escapeHtml(value) {
        return String(value === null || value === undefined ? '' : value)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }

    function skillList(requiredSkills) {
        if (!requiredSkills) return '';
        const chips = String(requiredSkills)
            .split(',')
            .map(skill => skill.trim())
            .filter(Boolean)
            .map(skill => `<li class="skill-chip">${escapeHtml(skill)}</li>`)
            .join('');
        return chips ? `<ul class="card-skills" aria-label="Skills required">${chips}</ul>` : '';
    }

    function displayRecommendations(recs) {
        if (!recs || recs.length === 0) {
            recommendationsDiv.innerHTML =
                '<p class="empty-state">No eligible internships matched your profile. ' +
                'Try adding more skills, choosing a different sector, or clearing the location preference.</p>';
            return;
        }

        recs.forEach(rec => {
            const card = document.createElement('div');
            card.className = 'recommendation-card';

            // Format the stipend for display, ensuring 0 is disclosed
            const stipendText = (rec.stipend !== null && rec.stipend !== '' && !isNaN(rec.stipend))
                ? `₹${new Intl.NumberFormat('en-IN').format(rec.stipend)} / month`
                : 'Not disclosed';
            const scoreText = (rec.score !== null && rec.score !== undefined && !isNaN(rec.score))
                ? Number(rec.score).toFixed(rec.mode === 'rule' ? 1 : 3)
                : 'N/A';
            const scoreLabel = rec.mode === 'rule' ? 'Rule score'
                : (rec.mode === 'ml' ? 'Text similarity' : 'Hybrid score');
            const explanation = rec.explanation
                ? `<p class="card-explanation">${escapeHtml(rec.explanation)}</p>`
                : '';

            card.innerHTML = `
                <div class="card-header">
                    <span class="sample-badge">Sample listing — not a real opening</span>
                    <h3>${escapeHtml(rec.title || 'Untitled internship')}</h3>
                    <p class="company">${escapeHtml(rec.company_name || 'Company not listed')}</p>
                </div>
                <div class="card-body">
                    ${skillList(rec.required_skills)}
                </div>
                <div class="card-details">
                    <p><strong>${scoreLabel}:</strong> <span class="card-score">${scoreText}</span></p>
                    <p><strong>Sector:</strong> ${escapeHtml(rec.sector || 'N/A')}</p>
                    <p><strong>Location:</strong> ${escapeHtml(rec.location || 'N/A')}</p>
                    <p class="card-stipend">${stipendText}</p>
                    <p><strong>Internship ID:</strong> ${escapeHtml(rec.internship_id || 'N/A')}</p>
                </div>
                ${explanation}
            `;
            recommendationsDiv.appendChild(card);
        });
    }
});
