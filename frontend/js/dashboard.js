// CareerLens — Dashboard JS
// Full app logic: file upload, API calls, UI rendering

const API_BASE = ''; // relative — served from same origin

// ── State ──────────────────────────────────────────────────────────────────
let state = {
  file: null,
  jd: '',
  analysis: null,
  bullets: [],
  selectedBullet: null,
};

// ── DOM refs ────────────────────────────────────────────────────────────────
const uploadZone     = document.getElementById('uploadZone');
const resumeFile     = document.getElementById('resumeFile');
const fileNameEl     = document.getElementById('fileName');
const jdTextarea     = document.getElementById('jobDescription');
const jdCharCount    = document.getElementById('jdCharCount');
const analyzeBtn     = document.getElementById('analyzeBtn');
const errorMsg       = document.getElementById('errorMsg');
const navStatus      = document.getElementById('navStatus');

const emptyState     = document.getElementById('emptyState');
const loadingState   = document.getElementById('loadingState');
const resultsContent = document.getElementById('resultsContent');

// ── Upload Zone ─────────────────────────────────────────────────────────────
uploadZone.addEventListener('click', () => resumeFile.click());

uploadZone.addEventListener('dragover', e => {
  e.preventDefault();
  uploadZone.classList.add('drag-over');
});
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('drag-over'));
uploadZone.addEventListener('drop', e => {
  e.preventDefault();
  uploadZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});

resumeFile.addEventListener('change', () => {
  if (resumeFile.files[0]) handleFile(resumeFile.files[0]);
});

function handleFile(file) {
  if (!file.name.toLowerCase().endsWith('.pdf')) {
    showError('Only PDF files are supported.');
    return;
  }
  state.file = file;
  fileNameEl.textContent = `✓ ${file.name}`;
  fileNameEl.classList.add('ready');
  uploadZone.classList.add('has-file');
  checkReady();
}

// ── JD Textarea ──────────────────────────────────────────────────────────────
jdTextarea.addEventListener('input', () => {
  state.jd = jdTextarea.value;
  jdCharCount.textContent = `${state.jd.length} characters`;
  checkReady();
});

function checkReady() {
  const ready = state.file && state.jd.length >= 50;
  analyzeBtn.disabled = !ready;
}

// ── Analyze ──────────────────────────────────────────────────────────────────
analyzeBtn.addEventListener('click', runAnalysis);

async function runAnalysis() {
  hideError();
  showLoading();

  const formData = new FormData();
  formData.append('resume', state.file);
  formData.append('job_description', state.jd);

  // Simulate loading steps
  const steps = ['step1','step2','step3','step4'];
  let stepIdx = 0;
  const stepInterval = setInterval(() => {
    if (stepIdx > 0) {
      document.getElementById(steps[stepIdx-1])?.classList.remove('active');
      document.getElementById(steps[stepIdx-1])?.classList.add('done');
    }
    if (stepIdx < steps.length) {
      document.getElementById(steps[stepIdx])?.classList.add('active');
      stepIdx++;
    }
  }, 2000);

  try {
    navStatus.textContent = 'Analyzing...';
    navStatus.classList.add('analyzing');
    analyzeBtn.classList.add('loading');
    analyzeBtn.querySelector('.btn-text').textContent = 'Analyzing...';

    const response = await fetch(`${API_BASE}/api/analyze`, {
      method: 'POST',
      body: formData,
    });

    clearInterval(stepInterval);

    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.detail || `Server error ${response.status}`);
    }

    state.analysis = await response.json();

    // Use full resume text from preview — API returns 800 chars, use all of it
    // and also scan the raw text if available
    state.bullets = extractBullets(
      (state.analysis.resume_text_preview || '') + '\n' +
      (state.analysis.resume_text || '')
    );
  

    renderResults(state.analysis);

    // Auto-fetch insights
    fetchInsights(state.analysis);

  } catch (err) {
    clearInterval(stepInterval);
    showError(err.message);
    showEmpty();
  } finally {
    navStatus.textContent = 'Analysis complete';
    navStatus.classList.remove('analyzing');
    analyzeBtn.classList.remove('loading');
    analyzeBtn.querySelector('.btn-text').textContent = 'Analyze Resume';
  }
}

// ── Render Results ────────────────────────────────────────────────────────────
function renderResults(data) {
  loadingState.classList.add('hidden');
  emptyState.classList.add('hidden');
  resultsContent.classList.remove('hidden');

  renderScore(data);
  renderHeatmap(data.skills);
  renderWeaknesses(data.weaknesses);
  renderBullets(state.bullets);
}

// Score Ring
function renderScore(data) {
  const score = data.match_score;
  const arc   = document.getElementById('scoreArc');
  const numEl = document.getElementById('scoreNumber');
  const label = document.getElementById('scoreLabel');
  const role  = document.getElementById('scoreRole');
  const badge = document.getElementById('scoreBadge');
  const sid   = document.getElementById('scoreId');

  // Color by score
  const color = score >= 75 ? '#00d4aa' :
                score >= 55 ? '#4cc9f0' :
                score >= 35 ? '#ffd166' : '#ff4d6d';
  arc.setAttribute('stroke', color);

  // Animate ring: circumference = 2π×58 ≈ 364.4
  const circ = 364.4;
  const offset = circ * (1 - score / 100);
  setTimeout(() => {
    arc.style.transition = 'stroke-dashoffset 1.2s cubic-bezier(.4,0,.2,1)';
    arc.style.strokeDashoffset = offset;
  }, 100);

  // Animate number
  animateNumber(numEl, 0, score, 1200);

  numEl.style.color = color;
  role.textContent  = data.job_title;
  sid.textContent   = `ID: ${data.analysis_id}`;

  const labelMap = { Excellent: 'excellent', Good: 'good', Fair: 'fair', Poor: 'poor' };
  badge.textContent = data.match_label;
  badge.className   = `score-badge ${labelMap[data.match_label] || 'fair'}`;
}

// Skill Heatmap
function renderHeatmap(skills) {
  const heatmap = document.getElementById('skillHeatmap');
  heatmap.innerHTML = '';

  skills.forEach((sk, i) => {
    const cell = document.createElement('div');
    cell.className = `skill-cell ${sk.status}`;
    cell.style.animationDelay = `${i * 40}ms`;

    const statusLabels = { strong: '✓ Strong', partial: '~ Partial', missing: '✗ Missing' };
    const pct = Math.round(sk.score * 100);

    cell.innerHTML = `
      <div class="skill-name">${escHtml(sk.skill)}</div>
      <div class="skill-bar">
        <div class="skill-bar-fill" style="width:${pct}%"></div>
      </div>
      <div class="skill-status">${statusLabels[sk.status]}</div>
    `;
    heatmap.appendChild(cell);
  });
}

// Weaknesses
function renderWeaknesses(weaknesses) {
  const list = document.getElementById('weaknessList');
  list.innerHTML = '';

  if (!weaknesses.length) {
    list.innerHTML = '<p style="color:var(--muted);font-size:.85rem">No major weaknesses detected. Great work!</p>';
    return;
  }

  weaknesses.forEach((w, i) => {
    const card = document.createElement('div');
    card.className = `weakness-card ${w.severity}`;
    card.style.animationDelay = `${i * 60}ms`;
    card.innerHTML = `
      <div class="weakness-header">
        <span class="weakness-section">${escHtml(w.section)}</span>
        <span class="severity-badge ${w.severity}">${w.severity.toUpperCase()}</span>
      </div>
      <div class="weakness-issue">${escHtml(w.issue)}</div>
      <div class="weakness-suggestion">💡 ${escHtml(w.suggestion)}</div>
    `;
    list.appendChild(card);
  });
}

// Bullet Points for Rewrite Tab
function renderBullets(bullets) {
  const list = document.getElementById('bulletList');
  list.innerHTML = '';

  if (!bullets.length) {
    list.innerHTML = '<p style="color:var(--muted);font-size:.85rem">No bullet points detected. Try adding bullet-point achievements to your resume.</p>';
    return;
  }

  bullets.slice(0, 12).forEach((bullet, i) => {
    const item = document.createElement('div');
    item.className = 'bullet-item';
    item.textContent = bullet;
    item.addEventListener('click', () => {
      document.querySelectorAll('.bullet-item').forEach(el => el.classList.remove('selected'));
      item.classList.add('selected');
      rewriteBullet(bullet);
    });
    list.appendChild(item);
  });
}

// Rewrite a bullet
async function rewriteBullet(bullet) {
  const panel   = document.getElementById('rewritePanel');
  const loading = document.getElementById('rewriteLoading');
  const result  = document.getElementById('rewriteResult');

  panel.classList.remove('hidden');
  loading.classList.remove('hidden');
  result.innerHTML = '';

  try {
    const res = await fetch(`${API_BASE}/api/rewrite`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        bullet_point: bullet,
        job_description: state.jd,
        context: 'Experience',
      }),
    });

    if (!res.ok) throw new Error('Rewrite failed');
    const data = await res.json();

    loading.classList.add('hidden');
    result.innerHTML = `
      <div class="before-after">
        <div class="ba-col before">
          <label>Before</label>
          <div class="ba-text">${escHtml(data.original)}</div>
        </div>
        <div class="ba-col after">
          <label>After</label>
          <div class="ba-text">${escHtml(data.rewritten)}</div>
        </div>
      </div>
      <div class="rewrite-reason">✦ ${escHtml(data.improvement_reason)}</div>
      <div class="rewrite-kws">
        ${data.impact_keywords.map(k => `<span class="kw-tag">${escHtml(k)}</span>`).join('')}
      </div>
      <button class="btn-copy" onclick="copyText('${escAttr(data.rewritten)}')">Copy Rewrite</button>
    `;
  } catch(err) {
    loading.classList.add('hidden');
    result.innerHTML = `<p style="color:var(--danger)">Rewrite failed: ${escHtml(err.message)}</p>`;
  }
}

// Career Insights
async function fetchInsights(analysis) {
  const missing = analysis.skills
    .filter(s => s.status === 'missing')
    .map(s => s.skill);

  try {
    const res = await fetch(`${API_BASE}/api/insights`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        missing_skills: missing,
        resume_score: analysis.match_score,
        job_title: analysis.job_title,
      }),
    });

    if (!res.ok) throw new Error('Insights fetch failed');
    const data = await res.json();
    renderInsights(data, analysis.match_score);
  } catch(err) {
    document.getElementById('insightsContainer').innerHTML =
      `<p style="color:var(--muted)">Could not load insights: ${escHtml(err.message)}</p>`;
  }
}

function renderInsights(data, currentScore) {
  const container = document.getElementById('insightsContainer');
  const projected = data.estimated_score_after_improvement;

  container.innerHTML = `
    <!-- Market insight -->
    <div class="insight-market">
      <div class="insight-market-label">🌐 Market Intelligence</div>
      <p>${escHtml(data.job_market_insight)}</p>
    </div>

    <!-- Score boost projection -->
    <div class="score-boost-bar">
      <div class="boost-label">Score projection after learning top skills</div>
      <div class="boost-track">
        <div class="boost-current" id="boostCurrent" style="width:0%"></div>
        <div class="boost-projected" id="boostProjected" style="width:0%"></div>
      </div>
      <div class="boost-numbers">
        <span class="boost-num now">Now: ${currentScore}</span>
        <span class="boost-num projected">After: ${projected} (+${(projected - currentScore).toFixed(1)})</span>
      </div>
    </div>

    <!-- Learning roadmap -->
    <div class="skills-roadmap">
      <div class="roadmap-title">🔥 Career Growth Roadmap</div>
      ${data.top_skills_to_learn.map((lp, i) => `
        <div class="skill-path-card" style="animation-delay:${i*100}ms">
          <div class="skill-path-header">
            <div class="skill-path-name">${escHtml(lp.skill)}</div>
            <span class="skill-path-priority">Priority #${lp.priority}</span>
          </div>
          <div class="skill-path-meta">
            <div class="path-meta-item">⏱ <strong>${escHtml(lp.estimated_time)}</strong></div>
            <div class="path-meta-item">📈 <strong>+${lp.impact_score.toFixed(0)}pts</strong> impact</div>
          </div>
          <div class="skill-path-resources">
            ${lp.resources.map(r => `<div class="resource-link">${escHtml(r)}</div>`).join('')}
          </div>
        </div>
      `).join('')}
    </div>

    <!-- Career tips -->
    <div class="career-tips">
      <div class="tips-title">💼 Career Growth Tips</div>
      ${data.career_growth_tips.map(t => `
        <div class="tip-item">${escHtml(t)}</div>
      `).join('')}
    </div>
  `;

  // Animate boost bars
  setTimeout(() => {
    document.getElementById('boostCurrent').style.width = `${currentScore}%`;
  }, 200);
  setTimeout(() => {
    document.getElementById('boostProjected').style.width = `${projected}%`;
  }, 400);
}

// ── Tabs ─────────────────────────────────────────────────────────────────────
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const tab = btn.dataset.tab;
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.add('hidden'));
    btn.classList.add('active');
    document.getElementById(`tab-${tab}`)?.classList.remove('hidden');
  });
});

// ── UI Helpers ────────────────────────────────────────────────────────────────
function showLoading() {
  emptyState.classList.add('hidden');
  resultsContent.classList.add('hidden');
  loadingState.classList.remove('hidden');
  // Reset steps
  document.querySelectorAll('.loader-step').forEach(s => {
    s.classList.remove('active', 'done');
  });
}

function showEmpty() {
  loadingState.classList.add('hidden');
  resultsContent.classList.add('hidden');
  emptyState.classList.remove('hidden');
}

function showError(msg) {
  errorMsg.textContent = msg;
  errorMsg.classList.add('visible');
}

function hideError() {
  errorMsg.classList.remove('visible');
  errorMsg.textContent = '';
}

function animateNumber(el, from, to, duration) {
  const start = performance.now();
  const update = (now) => {
    const t = Math.min((now - start) / duration, 1);
    const ease = 1 - Math.pow(1 - t, 3);
    el.textContent = Math.round(from + (to - from) * ease);
    if (t < 1) requestAnimationFrame(update);
  };
  requestAnimationFrame(update);
}

function extractBullets(text) {
  const lines = text.split('\n');
  const bullets = [];
  // Match any unicode bullet, dash, arrow, or numbered list at line start
  const bulletRe = /^[\u2022\u2023\u25E6\u2043\u2219\u25AA\u25AB\u25CF\u25CB\u2014\u2013\u2012\u2010\-\*\u2192\u25BA\u25B8▪◦○•–—→►]\s*/;
  for (const line of lines) {
    const s = line.trim();
    if (!s) continue;
    // Check bullet markers (including unicode variants)
    if (bulletRe.test(s)) {
      const clean = s.replace(bulletRe, '').trim();
      if (clean.length > 20) bullets.push(clean);
    }
    // Check numbered lists: 1. or 1)
    else if (/^\d{1,2}[.)]\s+\w/.test(s)) {
      const clean = s.replace(/^\d{1,2}[.)]\s+/, '').trim();
      if (clean.length > 20) bullets.push(clean);
    }
    // Catch lines starting with a special/non-ASCII char that looks like a bullet
    else if (s.charCodeAt(0) > 127 && s.charCodeAt(0) < 10000 && s[1] === ' ') {
      const clean = s.slice(2).trim();
      if (clean.length > 20) bullets.push(clean);
    }
  }
  return bullets;
}

function escHtml(str) {
  return String(str)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
    .replace(/"/g,'&quot;').replace(/'/g,'&#39;');
}

function escAttr(str) {
  return String(str).replace(/'/g, "\\'").replace(/\n/g, ' ');
}

function copyText(text) {
  navigator.clipboard.writeText(text).then(() => {
    const btn = document.querySelector('.btn-copy');
    if (btn) { btn.textContent = 'Copied!'; setTimeout(() => btn.textContent = 'Copy Rewrite', 2000); }
  });
}