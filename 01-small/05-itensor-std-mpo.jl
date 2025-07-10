
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

function electronic_structure(N, h1e, g2e, ecore)::MPO
    os = OpSum{Float64}()
    @time "\tConstructing OpSum" let
        for i in 1:N
            for j in 1:N
                abs(h1e[i, j]) < 1E-14 && continue
                os .+= h1e[i, j], "Cdagup", i, "Cup", j
                os .+= h1e[i, j], "Cdagdn", i, "Cdn", j
            end
        end
        for i in 1:N
            for j in 1:N
                for k in 1:N
                    for l in 1:N
                        abs(g2e[i, j, k, l]) < 1E-14 && continue
                        if (i != k && j != l)
                            os .+= 0.5 * g2e[i, j, k, l], "Cdagup", i, "Cdagup", k, "Cup", l, "Cup", j
                            os .+= 0.5 * g2e[i, j, k, l], "Cdagdn", i, "Cdagdn", k, "Cdn", l, "Cdn", j
                        end
                        os .+= 0.5 * g2e[i, j, k, l], "Cdagup", i, "Cdagdn", k, "Cdn", l, "Cup", j
                        os .+= 0.5 * g2e[i, j, k, l], "Cdagdn", i, "Cdagup", k, "Cup", l, "Cdn", j
                    end
                end
            end
        end
        os .+= ecore, "I", 1
    end
    sites = siteinds("Electron", N; conserve_qns=true)
    return @time "\tConstrucing MPO" MPO(os, sites)
end

data = npzread("00-ints.npz")

ncas = data["ncas"]
h1e = data["h1e"]
g2e = data["g2e"]
ecore = data["ecore"]
n_elec = data["n_elec"]

@time "Total" mpo = electronic_structure(ncas, h1e, g2e, ecore)
println("The maximum bond dimension is $(maxlinkdim(mpo))")

save_mpo("05-itensor-std-mpo.json", mpo)
