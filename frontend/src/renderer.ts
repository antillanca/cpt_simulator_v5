export interface Particle {
    x: number;
    y: number;
    vx?: number;
    vy?: number;
    color?: string;
}

export class Renderer {
    private canvas: HTMLCanvasElement;
    private ctx: CanvasRenderingContext2D;
    private particles: Particle[] = [];

    constructor(canvasId: string) {
        this.canvas = document.getElementById(canvasId) as HTMLCanvasElement;
        this.ctx = this.canvas.getContext('2d')!;
        this.resize();
        window.addEventListener('resize', () => this.resize());
    }

    private resize() {
        const container = this.canvas.parentElement;
        if (container) {
            this.canvas.width = container.clientWidth;
            this.canvas.height = container.clientHeight;
        }
    }

    public updateState(state: any) {
        // For now, let's assume the state is a single particle or a list
        if (state.particle) {
            this.particles = [state.particle];
        } else if (Array.isArray(state.particles)) {
            this.particles = state.particles;
        } else if (state.x !== undefined && state.y !== undefined) {
            this.particles = [state as Particle];
        }
    }

    public clear() {
        // Soft trail effect
        this.ctx.fillStyle = 'rgba(5, 7, 10, 0.2)';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Grid lines (subtle)
        this.drawGrid();
    }

    private drawGrid() {
        this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.03)';
        this.ctx.lineWidth = 1;
        const step = 50;
        
        for (let x = 0; x < this.canvas.width; x += step) {
            this.ctx.beginPath();
            this.ctx.moveTo(x, 0);
            this.ctx.lineTo(x, this.canvas.height);
            this.ctx.stroke();
        }
        
        for (let y = 0; y < this.canvas.height; y += step) {
            this.ctx.beginPath();
            this.ctx.moveTo(0, y);
            this.ctx.lineTo(this.canvas.width, y);
            this.ctx.stroke();
        }
    }

    public render() {
        this.clear();
        
        this.particles.forEach(p => {
            this.ctx.beginPath();
            const gradient = this.ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, 10);
            gradient.addColorStop(0, '#00f2ff');
            gradient.addColorStop(1, 'rgba(0, 242, 255, 0)');
            
            this.ctx.fillStyle = gradient;
            this.ctx.arc(p.x, p.y, 10, 0, Math.PI * 2);
            this.ctx.fill();
            
            // Core
            this.ctx.beginPath();
            this.ctx.fillStyle = '#fff';
            this.ctx.arc(p.x, p.y, 2, 0, Math.PI * 2);
            this.ctx.fill();
        });
    }

    public start() {
        const loop = () => {
            this.render();
            requestAnimationFrame(loop);
        };
        requestAnimationFrame(loop);
    }
}
