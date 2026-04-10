#!/usr/bin/env julia
#
# ITensor cross-reference for the Tindall N=127 kicked-Ising benchmark.
#
# Runs the same Floquet circuit Huoma executes in D.5.1–D.5.3, using
# ITensors.jl + ITensorNetworks.jl on the same Eagle 127q spanning tree
# (read from the committed golden file), at the same truncation budget.
# Output: a CSV of per-qubit ⟨Z⟩ at each depth checkpoint, with the
# same schema Huoma's test suite reads.
#
# This script is executed OFFLINE on a developer machine, not in CI.
# Its output is committed as data/z_expectations.csv and consumed by
# the Rust-side assertion in tests/ttn_tindall_127.rs.
#
# Usage:
#   cd external/itensor_ref
#   julia --project=. -e 'using Pkg; Pkg.instantiate()'   # first time
#   julia --project=. kim_heavy_hex.jl                     # run
#
# Requires Julia ≥ 1.10.

using JSON3
using CSV
using DataFrames
using ITensors
using ITensorNetworks

# ──────────────────────────────────────────────────────────────────────
# 1. Load the Eagle 127q topology from Huoma's golden file.
# ──────────────────────────────────────────────────────────────────────

const TOPOLOGY_PATH = joinpath(@__DIR__, "..", "..", "tests", "golden", "ibm_eagle_127.json")

function load_topology(path)
    raw = JSON3.read(read(path, String))
    n_qubits = raw.n_qubits
    tree_edges = [(e[1] + 1, e[2] + 1) for e in raw.tree_edges]  # 0-indexed → 1-indexed
    non_tree_edges = [(e[1] + 1, e[2] + 1) for e in raw.non_tree_edges]
    return n_qubits, tree_edges, non_tree_edges
end

# ──────────────────────────────────────────────────────────────────────
# 2. Build the ITN on the spanning tree.
# ──────────────────────────────────────────────────────────────────────

function build_tree_graph(n, tree_edges)
    g = NamedGraph(1:n)
    for (a, b) in tree_edges
        add_edge!(g, a => b)
    end
    return g
end

# ──────────────────────────────────────────────────────────────────────
# 3. KIM Floquet step — Tindall's exact convention.
# ──────────────────────────────────────────────────────────────────────

# θ_J = π/4 (ZZ per edge), θ_h = 0.8 (X kick per qubit).
# Unitary: U = (∏ exp(+i π/4 Z⊗Z)) · (∏ exp(-i θ_h/2 X))
const THETA_J = π / 4
const THETA_H = 0.8

function zz_gate(s1, s2, theta)
    # exp(+i θ Z⊗Z) — note positive sign per Tindall convention.
    # Diagonal: |00⟩,|11⟩ get exp(+iθ), |01⟩,|10⟩ get exp(-iθ).
    g = ITensor(ComplexF64, s1, s2, s1', s2')
    for v1 in dim(s1):-1:1, v2 in dim(s2):-1:1
        zz_eigenvalue = (v1 == v2) ? 1 : -1
        phase = exp(im * theta * zz_eigenvalue)
        g[s1 => v1, s2 => v2, s1' => v1, s2' => v2] = phase
    end
    return g
end

function rx_gate(s, theta)
    # exp(-i θ/2 X)
    c = cos(theta / 2)
    sn = sin(theta / 2)
    g = ITensor(ComplexF64, s, s')
    g[s => 1, s' => 1] = c
    g[s => 1, s' => 2] = -im * sn
    g[s => 2, s' => 1] = -im * sn
    g[s => 2, s' => 2] = c
    return g
end

# ──────────────────────────────────────────────────────────────────────
# 4. Main: run and emit CSV.
# ──────────────────────────────────────────────────────────────────────

function main()
    n_qubits, tree_edges, non_tree_edges = load_topology(TOPOLOGY_PATH)
    @info "Loaded Eagle topology" n_qubits n_tree=length(tree_edges) n_non_tree=length(non_tree_edges)

    # Coupling edges = tree ∪ non-tree.
    coupling_edges = vcat(tree_edges, non_tree_edges)

    # Build the tree-tensor-network graph.
    g = build_tree_graph(n_qubits, tree_edges)
    sites = siteinds("S=1/2", g)

    # Initial state: |↑↑…↑⟩ = all spin-up = |0…0⟩ in computational basis.
    psi = TTN(sites, v -> "Up")

    # Truncation parameters — match Huoma's run.
    chi_max = parse(Int, get(ENV, "CHI_MAX", "8"))
    n_steps = parse(Int, get(ENV, "N_STEPS", "20"))
    depth_checkpoints = parse.(Int, split(get(ENV, "DEPTHS", "5,10,20"), ","))

    @info "Running KIM" chi_max n_steps depth_checkpoints

    # Collect results.
    results = DataFrame(qubit=Int[], depth=Int[], z=Float64[])

    for step in 1:n_steps
        # ZZ layer on every coupling edge.
        for (a, b) in coupling_edges
            gate = zz_gate(sites[a], sites[b], THETA_J)
            psi = apply(gate, psi; maxdim=chi_max, cutoff=1e-16)
        end

        # Rx kick on every qubit.
        for v in 1:n_qubits
            gate = rx_gate(sites[v], THETA_H)
            psi = apply(gate, psi; maxdim=chi_max, cutoff=1e-16)
        end

        if step in depth_checkpoints
            @info "  depth=$step: computing ⟨Z⟩ for all qubits"
            for v in 1:n_qubits
                z = expect(psi, "Sz", v) * 2  # ITensor Sz = ±0.5; we want ±1
                push!(results, (qubit=v - 1, depth=step, z=z))  # 0-indexed qubit
            end
        end
    end

    outpath = joinpath(@__DIR__, "data", "z_expectations.csv")
    CSV.write(outpath, results)
    @info "Wrote $outpath ($(nrow(results)) rows)"
end

main()
