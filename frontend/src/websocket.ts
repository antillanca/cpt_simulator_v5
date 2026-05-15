export class WSClient {
    private socket: WebSocket | null = null;
    private url: string;
    private onMessageCallback: (data: any) => void;
    private onStatusCallback: (status: string) => void;

    constructor(url: string, onMessage: (data: any) => void, onStatus: (status: string) => void) {
        this.url = url;
        this.onMessageCallback = onMessage;
        this.onStatusCallback = onStatus;
    }

    public connect() {
        try {
            this.socket = new WebSocket(this.url);
            
            this.socket.onopen = () => {
                this.onStatusCallback('CONNECTED');
                console.log('WS Connected');
            };

            this.socket.onmessage = (event) => {
                const data = JSON.parse(event.data);
                this.onMessageCallback(data);
            };

            this.socket.onclose = () => {
                this.onStatusCallback('DISCONNECTED');
                console.log('WS Disconnected');
                // Auto-reconnect after 3 seconds
                setTimeout(() => this.connect(), 3000);
            };

            this.socket.onerror = (error) => {
                this.onStatusCallback('ERROR');
                console.error('WS Error:', error);
            };
        } catch (e) {
            console.error('Connection failed:', e);
            this.onStatusCallback('ERROR');
        }
    }

    public send(data: any) {
        if (this.socket && this.socket.readyState === WebSocket.OPEN) {
            this.socket.send(JSON.stringify(data));
        }
    }
}
