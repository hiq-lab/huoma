//! Solve feedback for Arn.
//!
//! Two transport modes:
//!
//! - **`grpc` feature enabled**: uses the generated tonic client from
//!   `proto/arn.proto` via [`ArnClient`]. One `ArnClient::connect` and
//!   the feedback flows as a typed protobuf message.
//!
//! - **Default (no feature)**: hand-rolled HTTP/1.1 POST of JSON to
//!   Arn's `/feedback` REST endpoint. No async runtime, no extra deps.
//!
//! Both modes are fire-and-forget: if Arn is unreachable, the feedback
//! is silently dropped. The simulation result is never affected.

use serde::Serialize;

// ── Generated proto types (gRPC mode) ───────────────────────────────

#[cfg(feature = "grpc")]
pub mod proto {
    #![allow(clippy::all)]
    include!("gen/arn.rs");
}

// ── Feedback payload (shared between both modes) ────────────────────

/// Per-island solve statistics.
#[derive(Debug, Clone, Serialize)]
pub struct IslandFeedback {
    pub n_qubits: usize,
    pub n_edges: usize,
    pub max_degree: usize,
    pub chi_per_edge: Vec<usize>,
    pub total_chi: usize,
    pub discarded_weight: f64,
    pub n_boundary_edges: usize,
    pub frequencies: Vec<f64>,
}

/// Aggregated feedback from a full ProjectedTtn solve.
#[derive(Debug, Clone, Serialize)]
pub struct SolveFeedback {
    pub n_qubits_total: usize,
    pub n_islands: usize,
    pub n_qubits_volatile: usize,
    pub volatile_fraction: f64,
    pub islands: Vec<IslandFeedback>,
    pub total_discarded_weight: f64,
    pub solve_ms: Option<f64>,
    /// Problem ID to correlate with an Arn analysis (optional).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub problem_id: Option<String>,
}

// ── gRPC transport ──────────────────────────────────────────────────

/// Arn gRPC client wrapper. Requires the `grpc` feature.
///
/// ```ignore
/// let client = ArnGrpcClient::connect("http://arn:4041").await?;
/// client.report_feedback(&feedback).await?;
/// ```
#[cfg(feature = "grpc")]
pub struct ArnGrpcClient {
    inner: proto::arn_client::ArnClient<tonic::transport::Channel>,
}

#[cfg(feature = "grpc")]
impl ArnGrpcClient {
    /// Connect to Arn's gRPC endpoint.
    pub async fn connect(addr: &str) -> Result<Self, tonic::transport::Error> {
        let inner = proto::arn_client::ArnClient::connect(addr.to_owned()).await?;
        Ok(Self { inner })
    }

    /// Send solve feedback to Arn. Returns the new threshold if Arn
    /// refined it, or None.
    pub async fn report_feedback(
        &mut self,
        fb: &SolveFeedback,
    ) -> Result<Option<f64>, tonic::Status> {
        let req = proto::FeedbackRequest {
            problem_id: fb.problem_id.clone().unwrap_or_default(),
            analysis_id: String::new(),
            status: "success".into(),
            energy: 0.0,
            solve_ms: fb.solve_ms.unwrap_or(0.0),
            n_qubits_total: fb.n_qubits_total as u32,
            n_islands: fb.n_islands as u32,
            n_qubits_volatile: fb.n_qubits_volatile as u32,
            volatile_fraction: fb.volatile_fraction,
            total_discarded_weight: fb.total_discarded_weight,
            islands: fb.islands.iter().map(|i| proto::IslandStats {
                n_qubits: i.n_qubits as u32,
                n_edges: i.n_edges as u32,
                max_degree: i.max_degree as u32,
                chi_per_edge: i.chi_per_edge.iter().map(|&c| c as u32).collect(),
                total_chi: i.total_chi as u32,
                discarded_weight: i.discarded_weight,
                n_boundary_edges: i.n_boundary_edges as u32,
                frequencies: i.frequencies.clone(),
            }).collect(),
        };

        let resp = self.inner.report_feedback(req).await?.into_inner();
        let new_threshold = if resp.new_threshold > 0.0 {
            Some(resp.new_threshold)
        } else {
            None
        };
        Ok(new_threshold)
    }
}

// ── HTTP/JSON transport (no feature needed) ─────────────────────────

/// Post feedback to Arn's REST `/feedback` endpoint via hand-rolled
/// HTTP/1.1 POST. Best-effort fire-and-forget.
pub fn post_feedback_http(arn_url: &str, feedback: &SolveFeedback) {
    let json = match serde_json::to_string(feedback) {
        Ok(j) => j,
        Err(_) => return,
    };

    let url = arn_url.trim_start_matches("http://");
    let (host_port, path) = match url.find('/') {
        Some(i) => (&url[..i], &url[i..]),
        None => (url, "/feedback"),
    };

    let request = format!(
        "POST {} HTTP/1.1\r\n\
         Host: {}\r\n\
         Content-Type: application/json\r\n\
         Content-Length: {}\r\n\
         Connection: close\r\n\
         \r\n\
         {}",
        path, host_port, json.len(), json
    );

    let stream = match std::net::TcpStream::connect(host_port) {
        Ok(s) => s,
        Err(_) => return,
    };
    let _ = stream.set_write_timeout(Some(std::time::Duration::from_millis(500)));
    use std::io::Write;
    let mut stream = stream;
    let _ = stream.write_all(request.as_bytes());
    let _ = stream.flush();
}

// ── Convenience: auto-dispatch based on feature ─────────────────────

/// Post feedback to Arn, using gRPC if the `grpc` feature is enabled
/// and a tokio runtime is available, otherwise falling back to HTTP/JSON.
///
/// This is the function `ProjectedTtn::post_feedback` calls.
pub fn post_feedback(arn_url: &str, feedback: &SolveFeedback) {
    #[cfg(feature = "grpc")]
    {
        // Try gRPC first. If a tokio runtime is active, use it.
        // If not (e.g., called from synchronous code), fall back to HTTP.
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            let url = arn_url.to_owned();
            let fb = feedback.clone();
            handle.spawn(async move {
                if let Ok(mut client) = ArnGrpcClient::connect(&url).await {
                    let _ = client.report_feedback(&fb).await;
                }
            });
            return;
        }
    }
    // Fallback: synchronous HTTP POST.
    post_feedback_http(arn_url, feedback);
}
