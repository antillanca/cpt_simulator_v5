#!/bin/bash
# stop_cpt_server.sh - Detener CPT Simulator v5 Server

echo "🛑 Deteniendo CPT Simulator v5..."

# Por puerto
fuser -k 8000/tcp 2>/dev/null

# Por proceso
pkill -9 -f "backend.main:app" 2>/dev/null

echo "✅ Server detenido"
