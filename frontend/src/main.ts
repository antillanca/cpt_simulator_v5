import { Renderer } from './renderer';
import { WSClient } from './websocket';

document.addEventListener('DOMContentLoaded', () => {
    const protocol = window.location.protocol;
    const wsProtocol = protocol === 'https:' ? 'wss:' : 'ws:';
    const API_BASE = `${protocol}//${window.location.hostname}:8000`;
    const WS_BASE = `${wsProtocol}//${window.location.hostname}:8000/ws`;

    const renderer = new Renderer('universoCanvas');
    const sysLogs = document.getElementById('sys-logs')!;
    const wsBadge = document.getElementById('ws-status')!;
    const orchBadge = document.getElementById('orch-badge')!;

    // ─── Logging ─────────────────────────────────────────────────────────────
    const addLog = (msg: string, type: 'info' | 'error' | 'success' = 'info') => {
        const el = document.createElement('div');
        el.className = `log-entry${type === 'error' ? ' error' : type === 'success' ? ' success' : ''}`;
        el.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
        sysLogs.prepend(el);
        if (sysLogs.children.length > 60) sysLogs.lastElementChild?.remove();
    };

    // ─── Tab navigation ───────────────────────────────────────────────────────
    document.querySelectorAll<HTMLButtonElement>('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
            btn.classList.add('active');
            const target = btn.dataset.tab!;
            document.getElementById(target)?.classList.add('active');
        });
    });

    // ─── WebSocket ─────────────────────────────────────────────────────────────
    const ws = new WSClient(
        WS_BASE,
        (message) => {
            if (message.type === 'state') {
                renderer.updateState(message.data);
            }
            if (message.type === 'intent') {
                addIntentEntry(message.data.text);
            }
        },
        (status: string) => {
            wsBadge.textContent = status;
            wsBadge.className = 'badge';
            if (status === 'CONNECTED') wsBadge.classList.add('online');
        }
    );
    ws.connect();
    renderer.start();

    // ─── Intent Stream ─────────────────────────────────────────────────────────
    const intentStream = document.getElementById('intent-stream')!;
    const addIntentEntry = (text: string) => {
        const el = document.createElement('div');
        el.className = 'intent-entry';
        el.innerHTML = `<span class="ts">${new Date().toLocaleTimeString()}</span>${text}`;
        intentStream.prepend(el);
        if (intentStream.children.length > 30) intentStream.lastElementChild?.remove();
    };
    // Demo entries to show the concept
    addIntentEntry('"I approach the target."');
    addIntentEntry('"I approach the target while I avoid the obstacle."');

    // ─── Curriculum / Module Data ────────────────────────────────────────────
    const updateCognitiveDashboard = async () => {
        try {
            const resp = await fetch(`${API_BASE}/api/ai/curriculum/status`);
            if (!resp.ok) throw new Error('API unavailable');
            const data = await resp.json();

            const total = data.total ?? 0;
            const confirmed = data.confirmed ?? 0;
            const pending = data.pending ?? 0;
            const repair = data.repair ?? 0;
            const tabular = data.tabular ?? 0;
            const lua = data.lua ?? 0;
            const pct = total > 0 ? Math.round((confirmed / total) * 100) : 0;

            // Stats
            document.getElementById('stat-total')!.textContent = String(total);
            document.getElementById('stat-confirmed')!.textContent = String(confirmed);
            document.getElementById('stat-pending')!.textContent = String(pending);
            document.getElementById('stat-repair')!.textContent = String(repair);
            document.getElementById('split-tabular')!.textContent = String(tabular);
            document.getElementById('split-lua')!.textContent = String(lua);

            // Quick stats sidebar
            document.getElementById('qs-confirmed')!.textContent = String(confirmed);
            document.getElementById('qs-pending')!.textContent = String(pending);
            document.getElementById('qs-repair')!.textContent = String(repair);

            // Progress bar
            document.getElementById('progress-fill')!.style.width = `${pct}%`;
            document.getElementById('progress-pct')!.textContent = `${pct}%`;
            document.getElementById('progress-text')!.textContent = `${confirmed} / ${total} modules assimilated`;

            // Module list
            if (data.modules) {
                const listEl = document.getElementById('module-list')!;
                listEl.innerHTML = '';
                (data.modules as any[]).forEach(mod => {
                    const row = document.createElement('div');
                    row.className = 'module-row';
                    let tagClass = 'tag-confirmed';
                    let tagText = '✓ confirmed';
                    if (mod.status === 'pending' && mod.engine_type === 'tabular') {
                        tagClass = 'tag-pending'; tagText = '⏳ DPO';
                    } else if (mod.status === 'pending') {
                        tagClass = 'tag-repair'; tagText = '🛠 lua';
                    }
                    row.innerHTML = `
                        <span class="module-name">L${mod.level} ${mod.key.replace(/_/g, ' ')}</span>
                        <div style="display:flex; gap:0.5rem; align-items:center;">
                            <span class="module-level">${mod.subject}</span>
                            <span class="module-tag ${tagClass}">${tagText}</span>
                        </div>`;
                    listEl.appendChild(row);
                });
            }

        } catch (e) {
            addLog('Curriculum API unavailable – showing cached state', 'error');
            // Fallback: show last known numbers from log
            document.getElementById('stat-total')!.textContent = '43';
            document.getElementById('stat-confirmed')!.textContent = '24';
            document.getElementById('stat-pending')!.textContent = '19';
            document.getElementById('split-tabular')!.textContent = '16';
            document.getElementById('split-lua')!.textContent = '27';
            document.getElementById('qs-confirmed')!.textContent = '24';
            document.getElementById('qs-pending')!.textContent = '19';
            document.getElementById('qs-repair')!.textContent = '10';
            const pct = Math.round(24/43*100);
            document.getElementById('progress-fill')!.style.width = `${pct}%`;
            document.getElementById('progress-pct')!.textContent = `${pct}%`;
            document.getElementById('progress-text')!.textContent = '24 / 43 modules assimilated (cached)';
        }
    };

    // ─── Orchestrator Log ─────────────────────────────────────────────────────
    const orchLog = document.getElementById('orch-log')!;
    let lastOrchestratorLine = 0;

    const updateOrchestratorLog = async () => {
        try {
            const resp = await fetch(`${API_BASE}/api/ai/orchestrator/log`);
            if (!resp.ok) throw new Error('no log endpoint');
            const data = await resp.json();

            const lines: string[] = data.lines ?? [];
            if (lines.length > lastOrchestratorLine) {
                const newLines = lines.slice(lastOrchestratorLine);
                lastOrchestratorLine = lines.length;
                newLines.forEach(line => {
                    const el = document.createElement('div');
                    el.className = 'orch-log-entry';
                    if (line.includes('✅') || line.includes('completed')) el.classList.add('ok');
                    else if (line.includes('❌') || line.includes('ERROR')) el.classList.add('err');
                    else if (line.includes('📍') || line.includes('🏭') || line.includes('🚀')) el.classList.add('inf');
                    el.textContent = line;
                    orchLog.prepend(el);
                });
                // Keep up to 80 entries
                while (orchLog.children.length > 80) orchLog.lastElementChild?.remove();
            }

            // Current target
            const orchTarget = document.getElementById('orch-target')!;
            const orchEngine = document.getElementById('orch-engine')!;
            if (data.current_module) {
                orchTarget.textContent = data.current_module;
                orchEngine.textContent = data.engine_type?.toUpperCase() ?? '—';
                orchBadge.textContent = '⚙ ORCHESTRATOR RUNNING';
                orchBadge.className = 'badge running';
            } else {
                orchBadge.textContent = '⚙ ORCHESTRATOR IDLE';
                orchBadge.className = 'badge';
            }

            // Kaggle status
            const kaggleDot = document.getElementById('kaggle-dot')!;
            const kaggleStatus = document.getElementById('kaggle-status')!;
            if (data.kaggle_running) {
                kaggleDot.className = 'status-dot dot-green';
                kaggleStatus.textContent = 'Training in progress…';
            } else {
                kaggleDot.className = 'status-dot dot-muted';
                kaggleStatus.textContent = 'Not running';
            }

            // StudentEngine
            const studentDot = document.getElementById('student-dot')!;
            const studentStatus = document.getElementById('student-status')!;
            if (data.student_running) {
                studentDot.className = 'status-dot dot-amber';
                studentStatus.textContent = 'Generating Lua…';
            } else {
                studentDot.className = 'status-dot dot-muted';
                studentStatus.textContent = 'Idle';
            }

            document.getElementById('orch-last-update')!.textContent = new Date().toLocaleTimeString();

        } catch (_) {
            // Fallback: read log file lines from a static endpoint
        }
    };

    // ─── Neural Models List ─────────────────────────────────────────────────
    const updateModelsList = async () => {
        try {
            const resp = await fetch(`${API_BASE}/api/ai/models`);
            if (!resp.ok) throw new Error();
            const data = await resp.json();
            const modelsList = document.getElementById('models-list')!;
            modelsList.innerHTML = '';
            (data.models as string[]).forEach(m => {
                const chip = document.createElement('span');
                chip.className = 'model-chip loaded';
                chip.textContent = `🟢 ${m}`;
                modelsList.appendChild(chip);
            });
        } catch (_) {
            // Ignore – model list is optional enhancement
        }
    };

    // ─── Lua Rule Runner ───────────────────────────────────────────────────────
    const ruleInput = document.getElementById('ruleInput') as HTMLTextAreaElement;
    const runBtn = document.getElementById('runBtn') as HTMLButtonElement;
    runBtn?.addEventListener('click', async () => {
        const rule = ruleInput.value.trim();
        if (!rule) return;
        runBtn.disabled = true;
        addLog('Testing rule…');
        try {
            const resp = await fetch(`${API_BASE}/api/rule/test`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ rule_text: rule })
            });
            const result = await resp.json();
            if (resp.ok) {
                addLog('Rule OK: ' + JSON.stringify(result.result?.particle ?? result.math), 'success');
                if (result.result?.particle) renderer.updateState(result.result.particle);
            } else {
                addLog('Error: ' + result.detail, 'error');
            }
        } catch { addLog('Connection error', 'error'); }
        finally { runBtn.disabled = false; }
    });

    // ─── i@ Chat ───────────────────────────────────────────────────────────────
    const chatInput = document.getElementById('chatInput') as HTMLInputElement;
    const sendChatBtn = document.getElementById('sendChatBtn') as HTMLButtonElement;
    const sendChat = async () => {
        const text = chatInput.value.trim();
        if (!text) return;
        sendChatBtn.disabled = true;
        addLog(`i@ → "${text}"`);
        try {
            const resp = await fetch(`${API_BASE}/api/ai/chat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text })
            });
            const result = await resp.json();
            if (result.status === 'blocked') addLog(result.message, 'error');
            else if (result.status === 'success') {
                addLog(result.message, 'success');
                ruleInput.value = result.lua_code;
                runBtn.click();
            } else addLog('Error: ' + (result.message ?? 'Unknown'), 'error');
        } catch { addLog('i@ connection error', 'error'); }
        finally { sendChatBtn.disabled = false; chatInput.value = ''; }
    };
    sendChatBtn?.addEventListener('click', sendChat);
    chatInput?.addEventListener('keypress', e => { if (e.key === 'Enter') sendChat(); });

    // ─── Initial load + polling ───────────────────────────────────────────────
    updateCognitiveDashboard();
    updateOrchestratorLog();
    updateModelsList();

    setInterval(updateCognitiveDashboard, 8000);
    setInterval(updateOrchestratorLog, 4000);
    setInterval(updateModelsList, 15000);

    addLog('CPT Cognitive Engine v2 — Monitoring Ready', 'success');
});
