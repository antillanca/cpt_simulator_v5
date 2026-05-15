#!/bin/bash
# start_cpt_server.sh - CPT Simulator v5 Server
# Arranca el servidor uvicorn en background de forma estable

cd /home/john/www/cpt_simulator_v5

# Verificar que Ollama esté corriendon
if ! pgrep -x "ollama" > /dev/null; then
    echo "⚠️  Ollama no está corriendo. Iniciando..."
    systemctl --user start ollama
    sleep 3
fi

# Liberar puerto si está ocupado
fuser -k 8000/tcp 2>/dev/null
sleep 1

# Arrancar servidor
echo "🚀 Iniciando CPT Simulator v5 en :8000..."
nohup python3 -m uvicorn backend.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    --timeout-keep-alive 120 \
    > /tmp/cpt_server.log 2>&1 &

SERVER_PID=$!
echo "✅ Server iniciado (PID=$SERVER_PID)"

# Esperar a que responda
for i in $(seq 1 10); do
    if curl -s http://localhost:8000/api/state/math > /dev/null 2>&1; then
        echo "✅ Server respondiendo en http://localhost:8000"
        echo "📊 Logs: tail -f /tmp/cpt_server.log"
        exit 0
    fi
    sleep 1
done

echo "⚠️  El servidor no responde. Verifica: tail -f /tmp/cpt_server.log"
