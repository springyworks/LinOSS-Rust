// Burn Profiler WebSocket Bridge
// 
// This server bridges LinossRust instrumentation data to the MaxGraph web interface
// by reading from the FIFO pipe and serving it via WebSocket

use std::{
    collections::HashMap,
    fs::OpenOptions,
    io::{BufRead, BufReader},
    net::SocketAddr,
    path::Path,
    sync::Arc,
    time::{Duration, Instant},
};

use serde::{Deserialize, Serialize};
use tokio::{
    net::{TcpListener, TcpStream},
    sync::{broadcast, Mutex},
    time::interval,
};
use tokio_tungstenite::{
    accept_async, tungstenite::protocol::Message,
};
use futures_util::{sink::SinkExt, stream::StreamExt};

// Data structures matching LinossRust instrumentation output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralInstrumentationData {
    pub timestamp: u64,
    pub simulation_time: f64,
    pub regions: Vec<RegionData>,
    pub system_stats: SystemStats,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionData {
    pub name: String,
    pub position: (f64, f64, f64),
    pub activity_magnitude: f64,
    pub velocity: (f64, f64, f64),
    pub dlinoss_state: DLinossStateData,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DLinossStateData {
    pub damping_coefficient: f64,
    pub oscillation_frequency: f64,
    pub energy_level: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemStats {
    pub total_operations: u64,
    pub memory_usage_mb: f64,
    pub fps: f64,
    pub coupling_strength: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct WebSocketMessage {
    #[serde(rename = "type")]
    message_type: String,
    data: Option<NeuralInstrumentationData>,
    timestamp: u64,
}

type Clients = Arc<Mutex<HashMap<SocketAddr, tokio_tungstenite::WebSocketStream<TcpStream>>>>;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔥 Burn Profiler WebSocket Bridge");
    println!("=================================");
    println!("Starting WebSocket server for MaxGraph neural dynamics visualization...");

    // Initialize shared state
    let (tx, _rx) = broadcast::channel::<NeuralInstrumentationData>(1000);
    let clients: Clients = Arc::new(Mutex::new(HashMap::new()));
    
    let tx_clone = tx.clone();
    let clients_clone = clients.clone();

    // Start FIFO pipe reader task
    let pipe_path = "/tmp/dlinoss_brain_pipe";
    tokio::spawn(async move {
        pipe_reader_task(pipe_path, tx_clone).await;
    });

    // Start WebSocket server
    let addr = "127.0.0.1:8080";
    let listener = TcpListener::bind(&addr).await?;
    println!("🌐 WebSocket server listening on: ws://{}", addr);
    println!("📡 Monitoring LinossRust instrumentation pipe: {}", pipe_path);
    println!("🎯 Open burn_profiler_maxgraph.html in browser to visualize");

    // Start metrics broadcast task
    let rx = tx.subscribe();
    tokio::spawn(async move {
        metrics_broadcast_task(clients_clone, rx).await;
    });

    // Accept WebSocket connections
    while let Ok((stream, addr)) = listener.accept().await {
        println!("🔌 New client connected: {}", addr);
        
        let clients_handle = clients.clone();
        tokio::spawn(async move {
            if let Err(e) = handle_connection(stream, addr, clients_handle).await {
                eprintln!("❌ Error handling connection {}: {}", addr, e);
            }
        });
    }

    Ok(())
}

async fn pipe_reader_task(pipe_path: &str, tx: broadcast::Sender<NeuralInstrumentationData>) {
    println!("📖 Starting FIFO pipe reader for: {}", pipe_path);
    
    let mut retry_count = 0;
    let max_retries = 5;
    
    loop {
        match read_from_pipe(pipe_path, &tx).await {
            Ok(_) => {
                println!("📖 Pipe reader completed normally");
                retry_count = 0;
            }
            Err(e) => {
                retry_count += 1;
                eprintln!("❌ Pipe reader error (attempt {}): {}", retry_count, e);
                
                if retry_count >= max_retries {
                    eprintln!("💀 Max retries reached, switching to simulation mode");
                    simulation_mode(&tx).await;
                    break;
                } else {
                    println!("⏳ Retrying in 2 seconds...");
                    tokio::time::sleep(Duration::from_secs(2)).await;
                }
            }
        }
    }
}

async fn read_from_pipe(
    pipe_path: &str,
    tx: &broadcast::Sender<NeuralInstrumentationData>,
) -> Result<(), String> {
    if !Path::new(pipe_path).exists() {
        return Err(format!("Pipe does not exist: {}", pipe_path));
    }

    println!("📡 Opening pipe for reading: {}", pipe_path);
    
    let file = OpenOptions::new().read(true).open(pipe_path)
        .map_err(|e| format!("Failed to open pipe: {}", e))?;
    let reader = BufReader::new(file);
    
    for line in reader.lines() {
        let line = line.map_err(|e| format!("Failed to read line: {}", e))?;
        if line.trim().is_empty() {
            continue;
        }

        // Parse JSON line
        match serde_json::from_str::<NeuralInstrumentationData>(&line) {
            Ok(data) => {
                if let Err(e) = tx.send(data) {
                    eprintln!("📡 Failed to broadcast data: {}", e);
                }
            }
            Err(e) => {
                eprintln!("⚠️ Failed to parse JSON: {} (line: {})", e, line.chars().take(100).collect::<String>());
            }
        }
    }

    Ok(())
}

async fn simulation_mode(tx: &broadcast::Sender<NeuralInstrumentationData>) {
    println!("🎮 Starting simulation mode - generating mock neural dynamics");
    
    let mut interval = interval(Duration::from_millis(100)); // 10 FPS
    let start_time = Instant::now();
    
    loop {
        interval.tick().await;
        
        let elapsed = start_time.elapsed().as_secs_f64();
        let mock_data = generate_mock_data(elapsed);
        
        if let Err(_) = tx.send(mock_data) {
            // No active receivers, continue anyway
        }
    }
}

fn generate_mock_data(time: f64) -> NeuralInstrumentationData {
    let regions = vec![
        RegionData {
            name: "prefrontal".to_string(),
            position: (
                (time * 0.3).sin() * 0.5,
                (time * 0.3).cos() * 0.5,
                (time * 0.7).sin() * 0.3,
            ),
            activity_magnitude: 0.5 + 0.3 * (time * 0.5).sin(),
            velocity: (
                (time * 0.8).cos() * 0.2,
                (time * 0.8).sin() * 0.2,
                (time * 1.2).cos() * 0.1,
            ),
            dlinoss_state: DLinossStateData {
                damping_coefficient: 0.1 + 0.05 * (time).sin(),
                oscillation_frequency: 2.0 + 0.5 * (time * 0.5).cos(),
                energy_level: (0.5 + 0.3 * (time * 0.5).sin()).powi(2),
            },
        },
        RegionData {
            name: "dmn".to_string(),
            position: (
                (time * 0.3 + 2.094).sin() * 0.5, // 120 degree phase shift
                (time * 0.3 + 2.094).cos() * 0.5,
                (time * 0.7 + 2.094).sin() * 0.3,
            ),
            activity_magnitude: 0.5 + 0.3 * (time * 0.5 + 2.094).sin(),
            velocity: (
                (time * 0.8 + 2.094).cos() * 0.2,
                (time * 0.8 + 2.094).sin() * 0.2,
                (time * 1.2 + 2.094).cos() * 0.1,
            ),
            dlinoss_state: DLinossStateData {
                damping_coefficient: 0.1 + 0.05 * (time + 2.094).sin(),
                oscillation_frequency: 2.0 + 0.5 * (time * 0.5 + 2.094).cos(),
                energy_level: (0.5 + 0.3 * (time * 0.5 + 2.094).sin()).powi(2),
            },
        },
        RegionData {
            name: "thalamus".to_string(),
            position: (
                (time * 0.3 + 4.188).sin() * 0.5, // 240 degree phase shift
                (time * 0.3 + 4.188).cos() * 0.5,
                (time * 0.7 + 4.188).sin() * 0.3,
            ),
            activity_magnitude: 0.5 + 0.3 * (time * 0.5 + 4.188).sin(),
            velocity: (
                (time * 0.8 + 4.188).cos() * 0.2,
                (time * 0.8 + 4.188).sin() * 0.2,
                (time * 1.2 + 4.188).cos() * 0.1,
            ),
            dlinoss_state: DLinossStateData {
                damping_coefficient: 0.1 + 0.05 * (time + 4.188).sin(),
                oscillation_frequency: 2.0 + 0.5 * (time * 0.5 + 4.188).cos(),
                energy_level: (0.5 + 0.3 * (time * 0.5 + 4.188).sin()).powi(2),
            },
        },
    ];

    NeuralInstrumentationData {
        timestamp: (std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis()) as u64,
        simulation_time: time,
        regions,
        system_stats: SystemStats {
            total_operations: (time * 100.0) as u64 + (time * 10.0).sin() as u64 * 10,
            memory_usage_mb: 45.0 + 5.0 * (time * 0.1).sin(),
            fps: 60.0 + 5.0 * (time * 0.2).cos(),
            coupling_strength: 0.1 + 0.05 * (time * 0.3).sin(),
        },
    }
}

async fn metrics_broadcast_task(
    clients: Clients,
    mut rx: broadcast::Receiver<NeuralInstrumentationData>,
) {
    println!("📊 Starting metrics broadcast task");
    
    while let Ok(data) = rx.recv().await {
        let message = WebSocketMessage {
            message_type: "neural_data".to_string(),
            data: Some(data),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
        };

        let json_message = match serde_json::to_string(&message) {
            Ok(json) => json,
            Err(e) => {
                eprintln!("❌ Failed to serialize message: {}", e);
                continue;
            }
        };

        // Send to all clients using tokio::sync::Mutex (async-aware)
        let mut disconnected_clients = Vec::new();
        
        {
            let mut clients_lock = clients.lock().await;
            for (addr, client) in clients_lock.iter_mut() {
                if let Err(e) = client.send(Message::Text(json_message.clone())).await {
                    eprintln!("❌ Failed to send to client {}: {}", addr, e);
                    disconnected_clients.push(*addr);
                }
            }

            // Remove disconnected clients
            for addr in disconnected_clients {
                clients_lock.remove(&addr);
                println!("🔌 Client disconnected: {}", addr);
            }
        }
    }
}

async fn handle_connection(
    stream: TcpStream,
    addr: SocketAddr,
    clients: Clients,
) -> Result<(), Box<dyn std::error::Error>> {
    let ws_stream = accept_async(stream).await?;
    
    // Add client to active connections
    {
        let mut clients_lock = clients.lock().await;
        clients_lock.insert(addr, ws_stream);
    }

    // Send welcome message
    {
        let mut clients_lock = clients.lock().await;
        if let Some(client) = clients_lock.get_mut(&addr) {
            let welcome_msg = WebSocketMessage {
                message_type: "welcome".to_string(),
                data: None,
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_millis() as u64,
            };
            
            let welcome_json = serde_json::to_string(&welcome_msg)?;
            let _ = client.send(Message::Text(welcome_json)).await;
        }
    }

    println!("✅ Client {} successfully connected and ready for neural data stream", addr);

    // Keep connection alive and handle incoming messages
    loop {
        tokio::time::sleep(Duration::from_secs(30)).await;
        
        // Send ping to keep connection alive
        {
            let mut clients_lock = clients.lock().await;
            if let Some(client) = clients_lock.get_mut(&addr) {
                if let Err(_) = client.send(Message::Ping(vec![])).await {
                    break; // Client disconnected
                }
            } else {
                break; // Client removed from map
            }
        }
    }

    Ok(())
}
