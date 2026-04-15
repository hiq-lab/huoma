//! Solve feedback for Arn's `/feedback` endpoint.
//!
//! After solving volatile islands, [`ProjectedTtn::collect_feedback`]
//! gathers per-island statistics (topology shape, frequencies, χ used,
//! discarded weight, wall time) into a [`SolveFeedback`] payload. This
//! can be posted to Arn so that the knowledge base learns which
//! sin(C/2) thresholds and χ allocations work on which problem classes.
//!
//! The feedback loop closes the Arn→Huoma→Arn cycle:
//!
//! ```text
//! Arn: problem → graph + frequencies → partition
//!   → Huoma: volatile islands → TTN solve → observables
//!     → Arn /feedback: island stats → knowledge base update
//! ```

use serde::Serialize;

/// Per-island solve statistics.
#[derive(Debug, Clone, Serialize)]
pub struct IslandFeedback {
    /// Number of qubits in this volatile island.
    pub n_qubits: usize,
    /// Number of edges in the island's spanning tree.
    pub n_edges: usize,
    /// Maximum degree of any node in the island.
    pub max_degree: usize,
    /// Per-edge χ that was used (length = n_edges).
    pub chi_per_edge: Vec<usize>,
    /// Total χ budget consumed (Σ chi_per_edge).
    pub total_chi: usize,
    /// Cumulative discarded weight after the solve (2-norm² truncation error).
    pub discarded_weight: f64,
    /// Number of boundary edges connecting this island to the stable exterior.
    pub n_boundary_edges: usize,
    /// Per-qubit frequencies (local ordering, length = n_qubits).
    pub frequencies: Vec<f64>,
}

/// Aggregated feedback from a full ProjectedTtn solve.
#[derive(Debug, Clone, Serialize)]
pub struct SolveFeedback {
    /// Total qubits in the full graph.
    pub n_qubits_total: usize,
    /// Number of volatile islands.
    pub n_islands: usize,
    /// Total volatile qubits (sum across islands).
    pub n_qubits_volatile: usize,
    /// Fraction of qubits that are volatile.
    pub volatile_fraction: f64,
    /// Per-island statistics.
    pub islands: Vec<IslandFeedback>,
    /// Total discarded weight across all islands.
    pub total_discarded_weight: f64,
    /// Wall time for the solve step in milliseconds (if measured by caller).
    pub solve_ms: Option<f64>,
}

/// Post feedback to Arn's `/feedback` endpoint.
///
/// This is a best-effort fire-and-forget POST. If Arn is unreachable
/// or returns an error, the feedback is silently dropped — Huoma's
/// simulation result is not affected.
///
/// Uses a raw TCP connection with a hand-written HTTP/1.1 POST to avoid
/// pulling in an async runtime or a heavy HTTP client crate. The payload
/// is small (a few KB of JSON) and the connection is closed immediately.
pub fn post_feedback(arn_url: &str, feedback: &SolveFeedback) {
    let json = match serde_json::to_string(feedback) {
        Ok(j) => j,
        Err(_) => return,
    };

    // Parse host:port from URL like "http://localhost:3000/feedback"
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
        path,
        host_port,
        json.len(),
        json
    );

    // Best-effort: if anything fails, silently return.
    let stream = match std::net::TcpStream::connect(host_port) {
        Ok(s) => s,
        Err(_) => return,
    };
    // Set a short timeout so we don't block the simulation.
    let _ = stream.set_write_timeout(Some(std::time::Duration::from_millis(500)));
    let _ = stream.set_read_timeout(Some(std::time::Duration::from_millis(500)));

    use std::io::Write;
    let mut stream = stream;
    let _ = stream.write_all(request.as_bytes());
    let _ = stream.flush();
    // Don't bother reading the response — fire and forget.
}
