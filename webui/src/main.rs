use axum::{
    extract::{State, Json},
    response::{sse::{Event, Sse}, Html, IntoResponse},
    routing::{get, post},
    Router,
};
use std::{convert::Infallible, net::SocketAddr};
use tokio::sync::broadcast;
use tokio_stream::{wrappers::BroadcastStream, StreamExt};
use serde::{Deserialize, Serialize};
use askama::Template;
use sysinfo::System;
use nvml_wrapper::Nvml;
use std::time::Duration;

#[derive(Serialize, Deserialize)]
struct SystemStatsPayload {
    cpu_usage_percent: f32,
    ram_used_mb: u64,
    ram_total_mb: u64,
    gpu_utilization_percent: u32,
    vram_used_mb: u64,
    vram_total_mb: u64,
}

#[derive(Clone)]
struct AppState {
    tx: broadcast::Sender<WebhookPayload>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct WebhookPayload {
    event: String,
    data: serde_json::Value,
}

#[derive(Template)]
#[template(path = "index.html")]
struct IndexTemplate {}

#[derive(Template)]
#[template(path = "message.html")]
struct MessageTemplate {
    sender: String,
    receiver: String,
    message_type: String,
    content: String,
}

#[derive(Template)]
#[template(path = "llm_call.html")]
struct LlmCallTemplate {
    agent_name: String,
    duration_str: String,
    duration_numeric: f64,
    total_tokens: i64,
    output_preview: String,
}

#[derive(Template)]
#[template(path = "agent_step.html")]
struct AgentStepTemplate {
    agent_name: String,
    duration_str: String,
    success: bool,
}

#[tokio::main]
async fn main() {
    let (tx, _) = broadcast::channel(1024);
    let state = AppState { tx: tx.clone() };

    tokio::spawn(async move {
        let mut sys = System::new_all();
        let nvml_opt = Nvml::init().ok();
        
        loop {
            sys.refresh_all();
            
            let mut cpu_tot = 0.0;
            let cpus = sys.cpus();
            if !cpus.is_empty() {
                for cpu in cpus {
                    cpu_tot += cpu.cpu_usage();
                }
                cpu_tot /= cpus.len() as f32;
            }

            let mut gpu_utilization_percent = 0;
            let mut vram_used_mb = 0;
            let mut vram_total_mb = 0;

            if let Some(ref nvml) = nvml_opt {
                if let Ok(device) = nvml.device_by_index(0) {
                    if let Ok(util) = device.utilization_rates() {
                        gpu_utilization_percent = util.gpu;
                    }
                    if let Ok(mem) = device.memory_info() {
                        vram_used_mb = mem.used / 1024 / 1024;
                        vram_total_mb = mem.total / 1024 / 1024;
                    }
                }
            }

            let stats = SystemStatsPayload {
                cpu_usage_percent: cpu_tot,
                ram_used_mb: sys.used_memory() / 1024 / 1024,
                ram_total_mb: sys.total_memory() / 1024 / 1024,
                gpu_utilization_percent,
                vram_used_mb,
                vram_total_mb,
            };

            let _ = tx.send(WebhookPayload {
                event: "system_stats".to_string(),
                data: serde_json::to_value(stats).unwrap_or_default(),
            });

            tokio::time::sleep(Duration::from_secs(1)).await;
        }
    });

    let app = Router::new()
        .route("/", get(index))
        .route("/stream", get(sse_stream))
        .route("/api/hook", post(webhook))
        .with_state(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], 3123));
    println!("Web UI listening on {}", addr);
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn index() -> impl IntoResponse {
    Html(IndexTemplate {}.render().unwrap())
}

async fn webhook(State(state): State<AppState>, Json(payload): Json<WebhookPayload>) -> impl IntoResponse {
    let _ = state.tx.send(payload);
    "OK"
}

async fn sse_stream(State(state): State<AppState>) -> Sse<impl tokio_stream::Stream<Item = Result<Event, Infallible>>> {
    let rx = state.tx.subscribe();
    let stream = BroadcastStream::new(rx).filter_map(|res| {
        match res {
            Ok(payload) => {
                // Serialize full payload as a hidden attribute for Frontend JS to intercept
                let raw_json = serde_json::to_string(&payload.data).unwrap_or_default();
                let html = match payload.event.as_str() {
                    "message" => {
                        let tmpl = MessageTemplate {
                            sender: payload.data["sender"].as_str().unwrap_or("").to_string(),
                            receiver: payload.data["receiver"].as_str().unwrap_or("").to_string(),
                            message_type: payload.data["message_type"].as_str().unwrap_or("").to_string(),
                            content: payload.data["content"].as_str().unwrap_or("").to_string(),
                        };
                        tmpl.render().ok()
                    },
                    "llm_call" => {
                        let tmpl = LlmCallTemplate {
                            agent_name: payload.data["agent_name"].as_str().unwrap_or("").to_string(),
                            duration_str: format!("{:.1}ms", payload.data["duration_ms"].as_f64().unwrap_or(0.0)),
                            duration_numeric: payload.data["duration_ms"].as_f64().unwrap_or(0.0),
                            total_tokens: payload.data["total_tokens"].as_i64().unwrap_or(0),
                            output_preview: payload.data["output_content"].as_str().unwrap_or("").to_string(),
                        };
                        tmpl.render().ok()
                    },
                    "agent_step" => {
                        let tmpl = AgentStepTemplate {
                            agent_name: payload.data["agent_name"].as_str().unwrap_or("").to_string(),
                            duration_str: format!("{:.1}ms", payload.data["duration_ms"].as_f64().unwrap_or(0.0)),
                            success: payload.data["success"].as_bool().unwrap_or(false),
                        };
                        tmpl.render().ok()
                    },
                    "system_stats" => {
                        let html_string = format!("<div class=\"hidden sse-event-block\" data-event=\"system_stats\" data-payload='{}'></div>", raw_json.replace("'", "&#39;"));
                        Some(html_string)
                    },
                    _ => None,
                };

                html.map(|h| Ok(Event::default().event(&payload.event).data(h)))
            },
            Err(_) => None,
        }
    });
    
    Sse::new(stream).keep_alive(axum::response::sse::KeepAlive::default())
}
