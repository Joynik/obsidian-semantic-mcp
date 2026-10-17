const DASHBOARD_TOKEN = document.querySelector('meta[name="osm-token"]').content;
const THEME_KEY = 'osm-dashboard-theme';
let dashboardInstance = null;
const customSelects = new Map();

function savedTheme() {
  try {
    const value = localStorage.getItem(THEME_KEY);
    if (value === 'light' || value === 'dark') return value;
  } catch (e) { /* Private browsing may block storage. */ }
  return window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
}

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  const next = theme === 'dark' ? 'light' : 'dark';
  document.getElementById('theme-icon').textContent = next === 'light' ? '☀' : '☾';
  document.getElementById('theme-label').textContent = t(next === 'light' ? 'Light theme' : 'Dark theme');
  document.getElementById('theme-toggle').setAttribute('aria-label', t(next === 'light' ? 'Switch to light theme' : 'Switch to dark theme'));
}

captureStaticText();
applyLanguage(savedLanguage());
applyTheme(savedTheme());
document.getElementById('language-select').addEventListener('change', event => {
  applyLanguage(event.target.value);
  applyTheme(document.documentElement.dataset.theme);
  try {
    localStorage.setItem(LANGUAGE_KEY, event.target.value);
  } catch (e) { /* Selection still works. */ }
  const empty = document.querySelector('#search-results .empty-state');
  if (empty) empty.textContent = t('No results found.');
  syncCustomSelects();
  fetchStats();
});
document.getElementById('theme-toggle').addEventListener('click', () => {
  const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
  applyTheme(next);
  try { localStorage.setItem(THEME_KEY, next); } catch (e) { /* Theme still changes for this tab. */ }
});

function timeAgo(iso) {
  if (!iso) return '—';
  const diff = Math.max(0, (Date.now() - new Date(iso).getTime()) / 1000);
  const [value, unit] = diff < 60 ? [Math.floor(diff), 'second']
    : diff < 3600 ? [Math.floor(diff / 60), 'minute']
      : diff < 86400 ? [Math.floor(diff / 3600), 'hour'] : [Math.floor(diff / 86400), 'day'];
  return new Intl.RelativeTimeFormat(document.documentElement.lang, { numeric: 'auto' }).format(-value, unit);
}

function dot(el, ok) {
  el.className = 'dot ' + (ok ? 'green' : 'red');
}

function toggleRecent() {
  document.getElementById('recent-list').classList.toggle('hidden');
  const button = document.getElementById('recent-toggle');
  button.classList.toggle('collapsed');
  button.setAttribute('aria-expanded', !button.classList.contains('collapsed'));
}

async function fetchStats() {
  try {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), 15000);
    const r = await fetch('/api/stats', { signal: ctrl.signal });
    clearTimeout(timer);
    const s = await r.json();
    document.getElementById('offline-banner').classList.add('hidden');
    const metrics = s.search_metrics;
    if (metrics) {
      document.getElementById('v-search-requests').textContent = metrics.available ? metrics.requests : '—';
      document.getElementById('v-search-duration').textContent = metrics.available ? (metrics.duration_ms / 1000).toFixed(2) + ' s' : '—';
      document.getElementById('d-search-outcomes').textContent =
        t('Results') + ': ' + metrics.results + ' · ' + t('Errors') + ': ' + metrics.errors;
      const backends = {};
      metrics.series.forEach(row => { backends[row.backend] = (backends[row.backend] || 0) + row.requests; });
      document.getElementById('d-search-backends').textContent = metrics.available
        ? (Object.entries(backends).map(([name, count]) => name + ': ' + count).join(' · ') || '0')
        : t('Unavailable');
    }

    dot(document.getElementById('dot-db'), s.db_ok);
    dot(document.getElementById('dot-ollama'), s.ollama_ok);
    dot(document.getElementById('dot-model'), s.model_loaded);

    document.getElementById('lbl-db').textContent =
      s.db_ok ? s.db_label : s.db_label + ' — ' + t('DOWN');
    document.getElementById('err-db').textContent =
      (!s.db_ok && s.db_error) ? s.db_error.split('\n')[0] : '';
    document.getElementById('err-db').title = document.getElementById('err-db').textContent;
    document.getElementById('lbl-ollama').textContent =
      s.ollama_ok ? 'Ollama' : 'Ollama — ' + t('DOWN');
    document.getElementById('btn-ollama').classList.toggle('hidden', s.ollama_ok || !s.can_start_ollama);
    document.getElementById('lbl-model').textContent =
      s.model_loaded ? s.embedding_model : t('Model — NOT LOADED');
    document.getElementById('vector-card').classList.toggle('hidden', s.db_label === 'Qdrant');
    document.getElementById('lbl-vector').textContent = s.vector_label;

    document.getElementById('v-indexed').textContent = s.indexed_count;
    document.getElementById('v-vault').textContent = s.vault_file_count;
    document.getElementById('v-gap').textContent = s.unindexed_count;
    document.getElementById('v-orphaned').textContent =
      s.orphaned_embeddings !== undefined ? s.orphaned_embeddings : '—';
    document.getElementById('v-dbsize').textContent = s.db_size_human;
    document.getElementById('v-last').textContent = timeAgo(s.last_indexed);
    document.getElementById('d-last').textContent = s.last_indexed
      ? new Date(s.last_indexed).toLocaleString(document.documentElement.lang) : '';
    document.getElementById('v-pgvec').textContent = 'v' + s.pgvector_version;
    document.getElementById('d-pgver').textContent = s.pg_version;

    const coverage = s.vault_file_count > 0
      ? Math.round(s.indexed_count / s.vault_file_count * 100) : 0;
    document.getElementById('d-indexed').textContent = coverage + '% ' + t('coverage');
    document.getElementById('d-vault').textContent = t('.md files in vault');

    const list = document.getElementById('recent-list');
    if (s.recent_notes.length === 0) {
      list.innerHTML = '<div class="recent-item"><span class="recent-path"></span></div>';
      list.querySelector('.recent-path').textContent = t('No notes indexed yet');
    } else {
      list.innerHTML = '';
      s.recent_notes.forEach(n => {
        const row = document.createElement('div');
        row.className = 'recent-item';
        const pathEl = document.createElement('span');
        pathEl.className = 'recent-path';
        pathEl.textContent = n.path;
        const timeEl = document.createElement('span');
        timeEl.className = 'recent-time';
        timeEl.textContent = n.indexed_at;
        row.appendChild(pathEl);
        row.appendChild(timeEl);
        list.appendChild(row);
      });
    }

    // Keep banner in sync with actual reindex state — prevents it getting
    // stuck visible after a server restart or page reload.
    if (!s.reindex_busy) {
      document.getElementById('indexing-banner').classList.add('hidden');
    }

    document.getElementById('footer').textContent =
      t('Last refresh') + ': ' + new Date().toLocaleTimeString(document.documentElement.lang) +
      ' · ' + t(document.getElementById('live-refresh').checked ? 'Live updates every 30s' : 'Live updates paused');

  } catch (e) {
    document.getElementById('offline-banner').classList.remove('hidden');
    const msg = e.name === 'AbortError' ? t('timed out') : e.message;
    document.getElementById('footer').textContent =
      t('Service unreachable') + ' (' + msg + ') — osm status';
    ['dot-db', 'dot-ollama', 'dot-model'].forEach(id => {
      document.getElementById(id).className = 'dot grey';
    });
  }
}

async function checkDashboard() {
  try {
    const response = await fetch('/api/health', { cache: 'no-store', signal: AbortSignal.timeout(2000) });
    if (!response.ok) throw new Error('Dashboard unavailable');
    const health = await response.json();
    if (dashboardInstance && health.instance !== dashboardInstance) {
      location.reload();
      return;
    }
    dashboardInstance = health.instance;
    document.getElementById('offline-banner').classList.add('hidden');
  } catch (e) {
    document.getElementById('offline-banner').classList.remove('hidden');
    document.getElementById('footer').textContent = t('Dashboard disconnected');
  }
}

function pollReindexDone(id) {
  fetch('/api/reindex/status').then(r => r.json()).then(d => {
    if (!d.busy) {
      const btn = document.getElementById(id);
      btn.disabled = false;
      btn.textContent = t(id === 'btn-rebuild' ? 'Safe Rebuild' : 'Re-index');
      document.getElementById('indexing-banner').classList.add('hidden');
      fetchStats();
    } else {
      setTimeout(() => pollReindexDone(id), 3000);
    }
  }).catch(() => {
    setTimeout(() => pollReindexDone(id), 5000);
  });
}

async function triggerReindex(full) {
  if (full && !confirm(t('Build a new index, then switch to it? The previous index will be archived.'))) return;
  const id = full ? 'btn-rebuild' : 'btn-reindex';
  const label = t(full ? 'Safe Rebuild' : 'Re-index');
  const btn = document.getElementById(id);
  btn.disabled = true;
  btn.textContent = t('Starting…');
  try {
    const r = await fetch(full ? '/api/reindex/full' : '/api/reindex', {
      method: 'POST',
      headers: { 'Authorization': 'Bearer ' + DASHBOARD_TOKEN },
    });
    const d = await r.json();
    if (d.ok) {
      btn.textContent = t('Running…');
      document.getElementById('indexing-banner').classList.remove('hidden');
      setTimeout(() => pollReindexDone(id), 3000);
    } else {
      btn.textContent = t('Failed') + ': ' + (d.message || '');
      setTimeout(() => { btn.disabled = false; btn.textContent = label; }, 5000);
    }
  } catch (e) {
    btn.textContent = t('Error');
    setTimeout(() => { btn.disabled = false; btn.textContent = label; }, 5000);
  }
}

async function startOllama() {
  const btn = document.getElementById('btn-ollama');
  btn.disabled = true;
  btn.textContent = t('Starting…');
  try {
    const r = await fetch('/api/ollama/start', {
      method: 'POST',
      headers: { 'Authorization': 'Bearer ' + DASHBOARD_TOKEN },
    });
    const d = await r.json();
    btn.textContent = t(d.ok ? 'Started' : 'Failed');
    setTimeout(fetchStats, 3000);
  } catch (e) {
    btn.textContent = t('Error');
  }
  setTimeout(() => { btn.disabled = false; btn.textContent = t('Start'); }, 5000);
}

async function testSearch() {
  const q = document.getElementById('search-query').value.trim();
  if (!q) return;
  const limit = document.getElementById('search-limit').value;
  const minSim = document.getElementById('search-min-sim').value;
  const mode = document.getElementById('search-mode').value;
  const vault = document.getElementById('search-vault').value;
  let url = `/api/search?q=${encodeURIComponent(q)}&limit=${limit}&min_similarity=${minSim}&mode=${mode}`;
  if (vault) url += `&vault=${encodeURIComponent(vault)}`;
  const res = await fetch(url);
  const data = await res.json();
  const div = document.getElementById('search-results');
  div.replaceChildren();
  if (!data.results || data.results.length === 0) {
    const empty = document.createElement('p');
    empty.className = 'empty-state';
    empty.textContent = t('No results found.');
    div.appendChild(empty);
    return;
  }
  data.results.forEach(result => {
    const card = document.createElement('div');
    card.className = 'search-result';
    const score = document.createElement('strong');
    score.textContent = (result.similarity * 100).toFixed(1) + '%';
    const path = document.createElement('code');
    path.textContent = result.path;
    const excerpt = document.createElement('p');
    excerpt.textContent = result.content.substring(0, 200) + '…';
    card.append(score, ' — ', path, excerpt);
    div.appendChild(card);
  });
}

async function fetchVaults() {
  try {
    const r = await fetch('/api/vaults');
    const d = await r.json();
    if (!d.vaults || d.vaults.length <= 1) return;
    const sel = document.getElementById('search-vault');
    d.vaults.forEach(v => {
      const opt = document.createElement('option');
      opt.value = v.name;
      opt.textContent = v.name;
      sel.appendChild(opt);
    });
    document.getElementById('vault-label').style.display = '';
    syncCustomSelects();
  } catch (e) { /* single-vault — keep hidden */ }
}

function closeCustomSelects() {
  customSelects.forEach(({ trigger, menu }) => {
    menu.hidden = true;
    trigger.setAttribute('aria-expanded', 'false');
  });
}

function syncCustomSelects() {
  customSelects.forEach(({ select, trigger, value, menu, label }) => {
    const selected = select.selectedOptions[0];
    value.textContent = selected ? selected.textContent : '';
    trigger.setAttribute('aria-label', t(label) + ': ' + value.textContent);
    menu.setAttribute('aria-label', t(label));
    menu.replaceChildren();
    Array.from(select.options).forEach((option, index) => {
      const item = document.createElement('button');
      item.type = 'button';
      item.className = 'select-option';
      item.setAttribute('role', 'option');
      item.setAttribute('aria-selected', String(option.selected));
      item.tabIndex = -1;
      item.textContent = option.textContent;
      item.addEventListener('click', () => {
        closeCustomSelects();
        select.value = option.value;
        select.dispatchEvent(new Event('change', { bubbles: true }));
        trigger.focus();
      });
      item.addEventListener('keydown', event => {
        const items = Array.from(menu.querySelectorAll('.select-option'));
        let next = index;
        if (event.key === 'ArrowDown') next = (index + 1) % items.length;
        else if (event.key === 'ArrowUp') next = (index - 1 + items.length) % items.length;
        else if (event.key === 'Home') next = 0;
        else if (event.key === 'End') next = items.length - 1;
        else if (event.key === 'Escape') {
          closeCustomSelects();
          trigger.focus();
          return;
        } else return;
        event.preventDefault();
        items[next].focus();
      });
      menu.appendChild(item);
    });
  });
}

function initCustomSelect(select, label) {
  const shell = document.createElement('div');
  shell.className = 'custom-select';
  const trigger = document.createElement('button');
  trigger.type = 'button';
  trigger.className = 'select-trigger';
  trigger.setAttribute('aria-haspopup', 'listbox');
  trigger.setAttribute('aria-expanded', 'false');
  const menu = document.createElement('div');
  menu.className = 'select-menu';
  menu.id = select.id + '-menu';
  menu.setAttribute('role', 'listbox');
  menu.setAttribute('aria-label', t(label));
  menu.hidden = true;
  trigger.setAttribute('aria-controls', menu.id);
  const value = document.createElement('span');
  value.className = 'select-value';
  const chevron = document.createElement('span');
  chevron.className = 'select-chevron';
  chevron.setAttribute('aria-hidden', 'true');
  trigger.append(value, chevron);
  shell.append(trigger, menu);
  select.insertAdjacentElement('afterend', shell);
  select.hidden = true;
  customSelects.set(select.id, { select, trigger, value, menu, label });
  trigger.addEventListener('click', () => {
    const wasOpen = !menu.hidden;
    closeCustomSelects();
    if (!wasOpen) {
      menu.hidden = false;
      trigger.setAttribute('aria-expanded', 'true');
    }
  });
  trigger.addEventListener('keydown', event => {
    if (!['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(event.key)) return;
    event.preventDefault();
    closeCustomSelects();
    menu.hidden = false;
    trigger.setAttribute('aria-expanded', 'true');
    const items = menu.querySelectorAll('.select-option');
    if (!items.length) return;
    const selected = Math.max(0, select.selectedIndex);
    const index = event.key === 'Home' ? 0 : event.key === 'End' ? items.length - 1
      : event.key === 'ArrowUp' ? Math.max(0, selected - 1) : Math.min(items.length - 1, selected + 1);
    items[index].focus();
  });
  select.addEventListener('change', syncCustomSelects);
}

initCustomSelect(document.getElementById('language-select'), 'Language');
initCustomSelect(document.getElementById('search-mode'), 'Mode');
initCustomSelect(document.getElementById('search-vault'), 'Vault');
syncCustomSelects();
document.addEventListener('pointerdown', event => {
  if (!event.target.closest('.custom-select')) closeCustomSelects();
});
document.addEventListener('keydown', event => {
  if (event.key === 'Escape') closeCustomSelects();
});

fetchStats();
fetchVaults();
checkDashboard();
function syncNavigation() {
  const current = location.hash || '#overview';
  document.querySelectorAll('.nav-link').forEach(link => {
    const selected = link.getAttribute('href') === current;
    link.classList.toggle('active', selected);
    if (selected) link.setAttribute('aria-current', 'location');
    else link.removeAttribute('aria-current');
  });
}
window.addEventListener('hashchange', syncNavigation);
syncNavigation();
document.getElementById('live-refresh').addEventListener('change', fetchStats);
setInterval(checkDashboard, 5000);
setInterval(() => {
  if (document.getElementById('live-refresh').checked) fetchStats();
}, 30000);
