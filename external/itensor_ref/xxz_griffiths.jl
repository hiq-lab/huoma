#!/usr/bin/env julia
#
# ITensor reference for the Track G bond-disordered XXZ shootout.
#
# Runs the same first-order-Trotter XXZ evolution that Huoma executes
# in src/xxz.rs::apply_xxz_step, on a 1D MPS chain, with combined
# bond-dim cap and discarded-weight truncation (ITensor's `maxdim` +
# `cutoff` knobs). The reference role: a second, structurally
# independent TEBD implementation that Huoma's three allocators
# (uniform-χ, sin(C/2), ε-truncated) are scored against.
#
# Executed OFFLINE on a developer machine. Output committed under
# `data/` and consumed by the Track G shootout driver on the Rust side.
#
# Usage:
#   cd external/itensor_ref
#   julia --project=. -e 'using Pkg; Pkg.instantiate()'   # first time
#   julia --project=. xxz_griffiths.jl <manifest.json>
#
# Convention bridge to Huoma:
#   - Huoma qubit indexing: qubit 0 is the MSB of `initial_index`
#   - ITensor site indexing: site 1 ↔ Huoma qubit 0 (left-to-right)
#   - Huoma `j_per_bond[i]` is the coupling on the bond between
#     qubits i and i+1, i.e. between ITensor sites (i+1) and (i+2)
#   - Huoma observable ⟨Z_q⟩ uses σ_z eigenvalues ±1; ITensor's
#     `expect(psi, "Sz", v)` returns the spin-1/2 expectation in ±0.5,
#     so every reported z is multiplied by 2 to match Huoma
#
# Requires Julia ≥ 1.10.

using JSON3
using ITensors
using ITensorMPS  # ITensors >= 0.7 split MPS/MPO ops into this package

# ──────────────────────────────────────────────────────────────────────
# Manifest schema (Rust-side writer in src/xxz.rs companion driver)
# ──────────────────────────────────────────────────────────────────────
#
# {
#   "n":              chain length (Int)
#   "delta":          XXZ anisotropy (Float64)
#   "dt":             Trotter step size (Float64)
#   "n_steps":        number of Trotter steps (Int)
#   "initial_index":  computational-basis initial-state index (UInt),
#                     MSB = qubit 0 (Huoma convention)
#   "j_per_bond":     length-(n-1) Vector{Float64} of bond couplings
#   "chi_cap":        maxdim bond-dim cap for ITensor `apply` (Int)
#   "epsilon":        cutoff (discarded-weight threshold) for `apply`
#                     (Float64; pass 0.0 to disable cutoff-truncation)
# }
# ──────────────────────────────────────────────────────────────────────

function load_manifest(path::AbstractString)
    raw = JSON3.read(read(path, String))
    n           = Int(raw.n)
    delta       = Float64(raw.delta)
    dt          = Float64(raw.dt)
    n_steps     = Int(raw.n_steps)
    initial_idx = UInt(raw.initial_index)
    j_per_bond  = Float64.(collect(raw.j_per_bond))
    chi_cap     = Int(raw.chi_cap)
    epsilon     = Float64(raw.epsilon)

    @assert length(j_per_bond) == n - 1 "j_per_bond length must be n-1 ($(n-1)), got $(length(j_per_bond))"
    @assert initial_idx < (UInt(1) << n) "initial_index out of range for n=$n"
    @assert chi_cap   >= 1   "chi_cap must be >= 1"
    @assert epsilon   >= 0.0 "epsilon must be >= 0"
    @assert n_steps   >= 0   "n_steps must be >= 0"
    @assert dt        != 0.0 "dt must be nonzero"

    return (; n, delta, dt, n_steps, initial_idx, j_per_bond, chi_cap, epsilon)
end

# ──────────────────────────────────────────────────────────────────────
# Initial state: product state matching Huoma's MSB convention.
# Huoma qubit q has bit (initial_index >> (n - 1 - q)) & 1.
# ITensor site k (1-indexed) ↔ Huoma qubit k-1.
# So site k's local state index = (initial_index >> (n - k)) & 1.
#  0 → "Up"  (Sz = +0.5, σz = +1)
#  1 → "Dn"  (Sz = -0.5, σz = -1)
# ──────────────────────────────────────────────────────────────────────

function product_state(sites, n::Int, initial_idx::UInt)
    states = Vector{String}(undef, n)
    for k in 1:n
        bit = (initial_idx >> (n - k)) & UInt(1)
        states[k] = bit == 0 ? "Up" : "Dn"
    end
    return MPS(sites, states)
end

# ──────────────────────────────────────────────────────────────────────
# XXZ bond gate as an ITensor.
#   H_bond_i = J_i (Sx_i Sx_{i+1} + Sy_i Sy_{i+1} + Δ Sz_i Sz_{i+1})
#   gate    = exp(-i dt H_bond_i)
# (Note: in the S=1/2 basis Sx=σx/2 etc, so this matches Huoma's
# (J/4)(σxσx + σyσy + Δσzσz) exactly.)
# ──────────────────────────────────────────────────────────────────────

function xxz_bond_gate(s1, s2, j::Float64, delta::Float64, dt::Float64)
    sx_sx = op("Sx", s1) * op("Sx", s2)
    sy_sy = op("Sy", s1) * op("Sy", s2)
    sz_sz = op("Sz", s1) * op("Sz", s2)
    h_bond = j * (sx_sx + sy_sy + delta * sz_sz)
    return exp(-1im * dt * h_bond)
end

# ──────────────────────────────────────────────────────────────────────
# Per-step observables and bond dimensions.
# ──────────────────────────────────────────────────────────────────────

function measure_all_z(psi)
    sz = expect(psi, "Sz")
    # ITensor returns ±0.5; Huoma wants ±1.
    return 2.0 .* collect(sz)
end

function measure_linkdims(psi)
    return collect(linkdims(psi))
end

# ──────────────────────────────────────────────────────────────────────
# Main driver.
# ──────────────────────────────────────────────────────────────────────

function main()
    if length(ARGS) != 1
        @error "Usage: julia --project=. xxz_griffiths.jl <manifest.json>"
        exit(1)
    end
    manifest_path = ARGS[1]
    @info "Loading manifest" manifest_path
    m = load_manifest(manifest_path)
    @info "Parameters" n=m.n delta=m.delta dt=m.dt n_steps=m.n_steps chi_cap=m.chi_cap epsilon=m.epsilon

    sites = siteinds("S=1/2", m.n; conserve_qns = false)
    psi   = product_state(sites, m.n, m.initial_idx)

    # Sanity: initial state ⟨Sz⟩ pattern must match the bits of initial_idx.
    @info "Initial linkdims" linkdims = collect(linkdims(psi))

    history = Vector{Vector{Float64}}()
    linkdim_per_step = Vector{Vector{Int}}()

    push!(history, measure_all_z(psi))
    push!(linkdim_per_step, measure_linkdims(psi))

    t0 = time()
    for step in 1:m.n_steps
        # First-order Trotter: sequential bonds 1..n-1 (Huoma bond i → ITensor sites (i+1, i+2))
        for i in 1:(m.n - 1)
            gate = xxz_bond_gate(sites[i], sites[i + 1], m.j_per_bond[i], m.delta, m.dt)
            psi  = apply(gate, psi; maxdim = m.chi_cap, cutoff = m.epsilon)
        end

        push!(history, measure_all_z(psi))
        push!(linkdim_per_step, measure_linkdims(psi))

        if step % max(1, fld(m.n_steps, 10)) == 0
            @info "step $step / $(m.n_steps) — max linkdim = $(maximum(linkdims(psi)))"
        end
    end
    wall_seconds = time() - t0
    @info "Run complete" wall_seconds

    # Output path: alongside the manifest, suffix ".itensor.json"
    out_path = replace(manifest_path, r"\.manifest\.json$" => ".itensor.json")
    if out_path == manifest_path
        # Manifest did not have the expected suffix; place output next to it.
        out_path = manifest_path * ".itensor.json"
    end

    output = (
        schema_version = 1,
        inputs = (
            n             = m.n,
            delta         = m.delta,
            dt            = m.dt,
            n_steps       = m.n_steps,
            initial_index = m.initial_idx,
            j_per_bond    = m.j_per_bond,
            chi_cap       = m.chi_cap,
            epsilon       = m.epsilon,
        ),
        history          = history,
        linkdims_per_step = linkdim_per_step,
        wall_seconds     = wall_seconds,
    )
    open(out_path, "w") do io
        JSON3.write(io, output)
    end
    @info "Wrote output" out_path
end

main()
