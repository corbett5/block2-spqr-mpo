
using NPZ
using JSON3
using ITensorMPOConstruction
using ITensorMPS
using ITensors

function save_mpo(filename, mpo::MPO)
    links = linkinds(mpo)
    sites = noprime(siteinds(first, mpo))
    r = []
    for i in 1:length(mpo)
        T = mpo[i]
        QS = inds(T)

        if i == 1
            @assert QS == (links[i], prime(sites[i]), dag(sites[i])) QS
        elseif i == length(mpo)
            @assert QS == (dag(links[end]), prime(sites[i]), dag(sites[i])) QS
        else
            @assert QS == (dag(links[i - 1]), links[i], prime(sites[i]), dag(sites[i])) QS
        end

        all_qns = []
        for j in 1:length(QS)
            qs = []
            for (qn, dim) in space(QS[j])
                qls = []
                for idx in qn
                    if (String(idx.name) != "")
                        push!(qls, (String(idx.name), idx.val[1]))
                    end
                end
                push!(qs, (qls, dim))
            end
            push!(all_qns, qs)
        end
        blocks = []
        for b in nzblocks(T)
            qns = Tuple([space(QS[ix])[x][1] for (ix, x) in enumerate(b)])

            if (i == 1)
                left_flux, right_flux = QN(), qns[1]
            elseif (i == length(mpo))
                left_flux, right_flux = qns[1], QN()
            else
                left_flux, right_flux = qns[1], qns[2]
            end

            local_flux = qns[end] - qns[end - 1]
            (left_flux + local_flux != right_flux) && println("Error at site $i, block $b: $left_flux, $local_flux, $right_flux")

            tag = Tuple([x - 1 for x in b])
            shape = size(T[b])
            arr = collect(Iterators.flatten(T[b]))
            push!(blocks, (tag, shape, arr))
        end
        push!(r, Dict("Q" => all_qns, "D" => blocks))
    end
    json_str = JSON3.write(r)
    open(filename, "w") do file
        write(file, json_str)
    end
end

function electronic_structure_opidsum(N, h1e, g2e, ecore)::MPO
    operatorNames = [
        [
            "I",
            "Cdn",
            "Cup",
            "Cdagdn",
            "Cdagup",
            "Ndn",
            "Nup",
            "Cup * Cdagup",
            "Cup * Nup",
            "Cdn * Cdagdn",
            "Cdn * Ndn",
            "Cup * Cdn",
            "Cup * Cdagdn",
            "Cup * Ndn",
            "Cdn * Cup",
            "Cdn * Cdagup",
            "Cdn * Nup",
            "Cdagup * Cup",
            "Cdagdn * Cdn",
            "Cdagup * Cdn",
            "Cdagup * Cdagdn",
            "Cdagup * Ndn",
            "Cdagdn * Cup",
            "Cdagdn * Cdagup",
            "Cdagdn * Nup",
            "Nup * Cdagup",
            "Nup * Nup",
            "Ndn * Cdagdn",
            "Ndn * Ndn",
            "Nup * Cdn",
            "Nup * Cdagdn",
            "Nup * Ndn",
            "Ndn * Cup",
            "Ndn * Cdagup",
            "Ndn * Nup",
        ] for _ in 1:N
    ]
    sites = siteinds("Electron", N; conserve_qns=true)
    op_cache_vec = to_OpCacheVec(sites, operatorNames)
  
    b = false
    a = true
  
    opC(k::Int, spin::Bool) = OpID(2 + spin, k)
    opCdag(k::Int, spin::Bool) = OpID(4 + spin, k)
  
    os = OpIDSum{4, Float64, Int64}(4*N^4 + 2*N^2, op_cache_vec)
    @time "\tConstructing OpIDSum" let
        for i in 1:N
            for j in 1:N
                abs(h1e[i, j]) < 1E-14 && continue
                add!(os, h1e[i, j], opCdag(i, a), opC(j, a))
                add!(os, h1e[i, j], opCdag(i, b), opC(j, b))
            end
        end
        for i in 1:N
            for j in 1:N
                for k in 1:N
                    for l in 1:N
                        abs(g2e[i, j, k, l]) < 1E-14 && continue
                        (i != k && j != l) && add!(os, 0.5 * g2e[i, j, k, l], opCdag(i, a), opCdag(k, a), opC(l, a), opC(j, a))
                        add!(os, 0.5 * g2e[i, j, k, l], opCdag(i, a), opCdag(k, b), opC(l, b), opC(j, a))
                        add!(os, 0.5 * g2e[i, j, k, l], opCdag(i, b), opCdag(k, a), opC(l, a), opC(j, b))
                        (i != k && j != l) && add!(os, 0.5 * g2e[i, j, k, l], opCdag(i, b), opCdag(k, b), opC(l, b), opC(j, b))
                    end
                end
            end
        end
        add!(os, ecore, OpID(1, 1))
    end
  
    return @time "\tConstructing MPO" MPO_new(
        os, sites; basis_op_cache_vec=op_cache_vec
    )
end

data = npzread("00-ints.npz")

ncas = data["ncas"]
h1e = data["h1e"]
g2e = data["g2e"]
ecore = data["ecore"]
n_elec = data["n_elec"]

@time "Total" mpo = electronic_structure_opidsum(ncas, h1e, g2e, ecore)
println("The maximum bond dimension is $(maxlinkdim(mpo))")

save_mpo("03-itensor-spqr-mpo.json", mpo)
