/* Live job search page: debounced queries, filters, pagination and error states.
 * All upstream calls go through /api/jobs, so the RapidAPI key never reaches
 * the browser. */
document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('job-search-form');
    const resultsContainer = document.getElementById('jobs-results-container');
    const resultsDiv = document.getElementById('jobs-results');
    const summaryP = document.getElementById('jobs-summary');
    const loadingDiv = document.getElementById('jobs-loading');
    const errorP = document.getElementById('jobs-error');
    const emptyP = document.getElementById('jobs-empty');
    const loadMoreBtn = document.getElementById('load-more-btn');

    const DEBOUNCE_MS = 450;
    const MIN_AUTO_QUERY = 3;

    let currentPage = 1;
    let nextCursor = '';
    let hasMore = false;
    let hasSearched = false;
    let inFlight = null;      // AbortController of the active request
    let debounceTimer = null;
    let seenIds = new Set();

    function escapeHtml(value) {
        return String(value === null || value === undefined ? '' : value)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }

    function currentParams(cursor) {
        const data = new FormData(form);
        const params = new URLSearchParams();
        params.set('q', (data.get('q') || '').trim());
        params.set('location', (data.get('location') || '').trim());
        params.set('country', data.get('country') || 'in');
        params.set('date_posted', data.get('date_posted') || 'all');
        params.set('work_mode', data.get('work_mode') || 'all');
        if (data.get('employment_type')) {
            params.set('employment_type', data.get('employment_type'));
        }
        // Profile fields set by the recommender form; used for scoring only.
        ['skills', 'qualification', 'sector'].forEach(key => {
            const value = (data.get(key) || '').trim();
            if (value) params.set(key, value);
        });
        const entryLevel = document.getElementById('entry_level');
        params.set('entry_level', entryLevel && entryLevel.checked ? '1' : '0');
        params.set('num_pages', '1');
        if (cursor) params.set('cursor', cursor);
        return params;
    }

    function showLoading(show) {
        loadingDiv.classList.toggle('hidden', !show);
    }

    function clearMessages() {
        errorP.classList.add('hidden');
        errorP.textContent = '';
        emptyP.classList.add('hidden');
        emptyP.textContent = '';
    }

    function relativeTime(iso) {
        if (!iso) return '';
        const posted = new Date(iso);
        if (isNaN(posted.getTime())) return '';
        const seconds = Math.max(0, Math.floor((Date.now() - posted.getTime()) / 1000));
        const units = [
            ['year', 31536000], ['month', 2592000], ['week', 604800],
            ['day', 86400], ['hour', 3600], ['minute', 60]
        ];
        for (const [label, size] of units) {
            if (seconds >= size) {
                const value = Math.floor(seconds / size);
                return `${value} ${label}${value === 1 ? '' : 's'} ago`;
            }
        }
        return 'just now';
    }

    function formatDate(iso) {
        if (!iso) return '';
        const date = new Date(iso);
        if (isNaN(date.getTime())) return '';
        return date.toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' });
    }

    function badges(job) {
        const items = [];
        if (job.is_remote === true) items.push('Remote');
        if (job.employment_type) items.push(job.employment_type);
        if (job.publisher) items.push(job.publisher);
        return items.map(text => `<span class="job-badge">${escapeHtml(text)}</span>`).join('');
    }

    function skillChips(skills) {
        if (!Array.isArray(skills) || skills.length === 0) return '';
        const chips = skills.map(skill => `<li class="skill-chip">${escapeHtml(skill)}</li>`).join('');
        return `<ul class="card-skills" aria-label="Skills listed in the posting">${chips}</ul>`;
    }

    function matchBadge(match) {
        if (!match) return '';
        const level = match.percent >= 70 ? 'strong' : (match.percent >= 40 ? 'partial' : 'weak');
        return `<span class="match-badge ${level}" title="${escapeHtml(match.explanation)}">`
            + `${match.percent}% match</span>`;
    }

    function matchedSkillChips(match) {
        if (!match || !match.matched_skills || match.matched_skills.length === 0) return '';
        const chips = match.matched_skills
            .map(skill => `<li class="skill-chip match-chip">${escapeHtml(skill)}</li>`).join('');
        return `<ul class="card-skills" aria-label="Skills from your profile found in this posting">${chips}</ul>`;
    }

    function buildCard(job) {
        const card = document.createElement('div');
        card.className = 'recommendation-card job-card';
        const match = job.match;

        const posted = relativeTime(job.posted_at);
        const postedLine = posted
            ? `<p class="job-posted">Posted ${escapeHtml(posted)}${job.posted_at ? ` &middot; ${escapeHtml(formatDate(job.posted_at))}` : ''}</p>`
            : '<p class="job-posted">Posting date not provided</p>';

        const location = job.location
            ? `<p><strong>Location:</strong> ${escapeHtml(job.location)}</p>`
            : '<p><strong>Location:</strong> Not specified</p>';

        const salary = job.salary
            ? `<p class="card-stipend">${escapeHtml(job.salary)}</p>` : '';

        const description = job.description
            ? `<p class="job-description" data-collapsed="true">${escapeHtml(job.description)}</p>`
            : '<p class="job-description">No description provided.</p>';

        const detailsButton = job.description && job.description.length > 220
            ? '<button type="button" class="link-btn job-details-toggle">Show full description</button>'
            : '';

        const applyCell = job.apply_link
            ? `<a class="apply-btn" href="${escapeHtml(job.apply_link)}" target="_blank"
                  rel="noopener noreferrer">Apply on ${escapeHtml(job.publisher || 'original site')}</a>`
            : '<span class="apply-btn apply-btn-disabled" aria-disabled="true">No application link provided</span>';

        const logo = job.company_logo
            ? `<img class="employer-logo" src="${escapeHtml(job.company_logo)}" alt="" loading="lazy"
                    onerror="this.style.display='none'">` : '';

        card.innerHTML = `
            <div class="card-header">
                <div class="badge-row">
                    <span class="sample-badge live-badge">Live listing</span>
                    ${matchBadge(match)}
                </div>
                <h3>${escapeHtml(job.title)}</h3>
                <p class="company">${logo}${escapeHtml(job.company)}</p>
            </div>
            <div class="card-body">
                <div class="job-badges">${badges(job)}</div>
                ${location}
                ${postedLine}
                ${matchedSkillChips(match)}
                ${skillChips(job.required_skills)}
                ${description}
                ${detailsButton}
            </div>
            <div class="card-details">
                ${salary}
                <p><strong>Employment type:</strong> ${escapeHtml(job.employment_type || 'Not specified')}</p>
            </div>
            ${match && match.explanation ? `<p class="card-explanation">${escapeHtml(match.explanation)}</p>` : ''}
            <div class="job-actions">${applyCell}</div>
        `;

        const toggle = card.querySelector('.job-details-toggle');
        if (toggle) {
            toggle.addEventListener('click', () => {
                const paragraph = card.querySelector('.job-description');
                const collapsed = paragraph.getAttribute('data-collapsed') === 'true';
                paragraph.setAttribute('data-collapsed', collapsed ? 'false' : 'true');
                toggle.textContent = collapsed ? 'Show less' : 'Show full description';
            });
        }
        return card;
    }

    function renderJobs(jobs, append) {
        if (!append) {
            resultsDiv.innerHTML = '';
            seenIds = new Set();
        }
        let added = 0;
        jobs.forEach(job => {
            const key = job.id || `${job.title}|${job.company}|${job.location}`;
            if (seenIds.has(key)) return;   // the same posting can repeat across pages
            seenIds.add(key);
            resultsDiv.appendChild(buildCard(job));
            added += 1;
        });
        return added;
    }

    function describeFilters(params) {
        const parts = [];
        if (params.get('q')) parts.push(`"${params.get('q')}"`);
        if (params.get('location')) parts.push(`in ${params.get('location')}`);
        return parts.join(' ') || 'all jobs';
    }

    async function runSearch({ append = false } = {}) {
        const params = currentParams(append ? nextCursor : '');

        if (!params.get('q') && !params.get('location')) {
            resultsContainer.classList.remove('hidden');
            resultsDiv.innerHTML = '';
            summaryP.textContent = '';
            loadMoreBtn.classList.add('hidden');
            clearMessages();
            errorP.textContent = 'Enter a job title, skill or location to search.';
            errorP.classList.remove('hidden');
            return;
        }

        if (inFlight) inFlight.abort();
        inFlight = new AbortController();

        hasSearched = true;
        resultsContainer.classList.remove('hidden');
        showLoading(true);
        if (!append) clearMessages();
        loadMoreBtn.classList.add('hidden');

        try {
            const response = await fetch('/api/jobs?' + params.toString(), {
                headers: { 'Accept': 'application/json' },
                signal: inFlight.signal
            });

            if (response.status === 401) {
                window.location.href = '/login?next=' + encodeURIComponent('/jobs');
                return;
            }

            const payload = await response.json().catch(() => null);

            if (!response.ok) {
                const message = (payload && payload.error) || 'The job search failed. Please try again.';
                throw new Error(message);
            }

            nextCursor = payload.next_cursor || '';
            hasMore = Boolean(payload.has_more) && Boolean(nextCursor);
            currentPage = append ? currentPage + 1 : 1;
            const added = renderJobs(payload.jobs || [], append);

            const hidden = payload.filtered_out || 0;
            const hiddenNote = hidden
                ? ` · ${hidden} role${hidden === 1 ? '' : 's'} hidden (experience required)`
                : '';
            summaryP.textContent = added === 0
                ? (hidden ? `${hidden} role${hidden === 1 ? '' : 's'} hidden — all required experience` : '')
                : `${describeFilters(params)} — showing ${resultsDiv.children.length} listing`
                  + `${resultsDiv.children.length === 1 ? '' : 's'}${payload.cached ? ' (cached)' : ''}`
                  + hiddenNote;

            if (added === 0 && !append) {
                emptyP.textContent = hidden
                    ? 'Every match for those filters asks for experience. Untick "Entry level only" '
                      + 'to include them, or widen the keyword or time window.'
                    : 'No jobs matched those filters. Try a broader keyword, '
                      + 'a different location, or a longer time window.';
                emptyP.classList.remove('hidden');
            } else if (added === 0 && append) {
                emptyP.textContent = 'No more jobs for this search.';
                emptyP.classList.remove('hidden');
            }

            loadMoreBtn.classList.toggle('hidden', !hasMore);
        } catch (error) {
            if (error.name === 'AbortError') return;
            showLoading(false);
            errorP.textContent = error.message;
            errorP.classList.remove('hidden');
            if (!append) {
                resultsDiv.innerHTML = '';
                summaryP.textContent = '';
            }
        } finally {
            showLoading(false);
            inFlight = null;
        }
    }

    form.addEventListener('submit', (event) => {
        event.preventDefault();
        if (debounceTimer) clearTimeout(debounceTimer);
        runSearch();
    });

    loadMoreBtn.addEventListener('click', () => {
        if (!hasMore) return;
        runSearch({ append: true });
    });

    // Filter dropdowns apply immediately; free-text fields debounce so a user
    // typing "developer" costs one request instead of nine.
    const entryLevelBox = document.getElementById('entry_level');
    if (entryLevelBox) {
        entryLevelBox.addEventListener('change', () => {
            if (debounceTimer) clearTimeout(debounceTimer);
            runSearch();
        });
    }

    form.querySelectorAll('select').forEach(select => {
        select.addEventListener('change', () => {
            if (debounceTimer) clearTimeout(debounceTimer);
            runSearch();
        });
    });

    // Arriving from the recommender with a profile pre-fills the form: run once so the
    // user sees matches immediately instead of an empty page.
    if ((document.getElementById('q').value || '').trim()) {
        runSearch();
    }

    ['q', 'location'].forEach(id => {
        const input = document.getElementById(id);
        input.addEventListener('input', () => {
            if (debounceTimer) clearTimeout(debounceTimer);
            debounceTimer = setTimeout(() => {
                const typed = (input.value || '').trim();
                const other = id === 'q'
                    ? (document.getElementById('location').value || '').trim()
                    : (document.getElementById('q').value || '').trim();
                if (!hasSearched) return;                       // no request on a blank page
                if (typed.length < MIN_AUTO_QUERY && !other) return;
                runSearch();
            }, DEBOUNCE_MS);
        });
    });
});
