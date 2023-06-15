#!/usr/bin/env julia
# -*- coding: utf-8 -*- 
# Created on Thu Mar  9 07:22:59 PM EST 2023 
# author: Ryan Hildebrandt, github.com/ryancahildebrandt

# imports
using WordTokenizers
using StatsBase
using Distances
using DataFrames
using Clustering
using UMAP
using Statistics
using Random 
using MLJText
using MLJ
using MLJFlux
using Flux
using EvoTrees
using MLJXGBoostInterface
using XGBoost

rng = Random.default_rng()
Random.seed!(rng, 42)
set_tokenizer(nltk_word_tokenize)

# data dict
function build_data_dict(dataset::DataFrame, dataset_name::String)
    out = Dict()
    out["name"] = dataset_name 
    out["data"] = dataset
    out["docs"] = dataset[!, "text"]
    out["labels"] = dataset[!, "label"]
    out["int"] = dataset[!, "int"]
    out["vocab"], out["coords"], out["dists"], out["clusts"] = docs_to_clusters(out["docs"])
    @info "Clustering with k=10"
    @time out["df_10"], out["gb_10"] = clusters_to_gb(out["clusts"], out["vocab"], out["dists"], out["coords"], 10)
    @info "Clustering with k=25"
    @time out["df_25"], out["gb_25"] = clusters_to_gb(out["clusts"], out["vocab"], out["dists"], out["coords"], 25)
    @info "Clustering with k=50"
    @time out["df_50"], out["gb_50"] = clusters_to_gb(out["clusts"], out["vocab"], out["dists"], out["coords"], 50)
    return out
end

# preprocessing & labels
function word_prep(word::Union{String, SubString{String}})
        word = lowercase(word)
        is_feature = contains(word, r"<feature_.+>")
        is_nt = word == "n't"
        is_vocab = word in _vocab
        if is_vocab | is_feature | is_nt
            out = word
        else
            out = "OOV"
        end
    return out
end

function doc_prep(doc::String)
    has_space = contains(doc, " ")
    new_doc = String[]
    split_doc = split(doc)
    if has_space
        for d in split_doc
            is_feature = contains(d, r"<feature_.+>")
            is_nt = d == "n't"        
            if is_feature | is_nt
                new_doc = push!(new_doc, word_prep(d))
            else
                new_doc = push!(new_doc, tokenize(d)...)
            end
        end
    else
        new_doc = [word_prep(doc)]
    end
    out = word_prep.(new_doc)
    return out
end

function str_to_int(labels)
    label_map = Dict(lab => int for (int,lab) in enumerate(unique(labels)))
    out = [label_map[i] for i in labels]
    return out
end

# embedding
function encode(doc::String)
    doc = doc_prep(doc)
    ind = [ind_dict[d] for d in doc]
    out = emb_df[ind, "emb"]
    return out
end

function docs_to_word_embs(docs::Vector{String})
    embs = encode.(docs)
    coords = reduce(hcat, vcat(embs...))
    @time out = center_coords(umap(coords, n_neighbors = 5))
    return out
end

function encode_bow(transformer::Union{TfidfTransformer, CountTransformer, BM25Transformer}, docs::Vector{String})
    mach = machine(transformer, docs, scitype_check_level = 0)
    MLJ.fit!(mach)
    @time mat = MLJ.transform(mach, doc_prep.(docs))
    out = Matrix(mat)
    return out
end

# vocab
function pull_vocab(docs::Vector{String})
    out = []
    for doc in docs
        doc = doc_prep(doc)
        push!(out, doc)
    end
    out = union(vcat(out...))
    return out
end

function combine_features(grouped_vocab::Vector{Vector{String}}, metric::Vector{Float64}, threshold::Number, feature_override::Vector)
    flat_vocab = vcat(grouped_vocab...)
    out = Dict(flat_vocab .=> flat_vocab)
    n = 0
    for (group, met) in zip(grouped_vocab, metric)
        if 0.0 < met < threshold
            n += 1
            for word in group
                out[word] = "<feature_$n>"
            end
        end
    end
    for word in feature_override
        out[word] = word
    end
    return out
end

function replace_features(docs::Vector{String}, feature_dict::Dict)
    mapped_docs = map.(x -> feature_dict[x], doc_prep.(docs))
    out = join.(mapped_docs, " ")
end

# cluster
function mean_eucl(x::Union{Vector, SubArray}, y::Union{Vector, SubArray})
    centroid = [mean(x), mean(y)]
    out = mean(colwise(Euclidean(), centroid, hcat(x, y)'))
    return out
end

function center_coords(x::Vector{Float64}, y::Vector{Float64})
    xc = x .- mean(x)
    yc = y .- mean(y)
    out =  hcat(xc, yc)'
    return out 
end

function center_coords(m::Matrix{Float64})
    x = m[1, :]
    y = m[2, :]
    xc = x .- mean(x)
    yc = y .- mean(y)
    out =  hcat(xc, yc)'
    return out 
end

function docs_to_clusters(docs::Vector{String})
    vocab = pull_vocab(docs)
    coords = docs_to_word_embs(vocab)
    dists = pairwise(Euclidean(), coords, dims = 2)
    clusts = hclust(dists)
    out = [vocab, coords, dists, clusts]
    return out
end

function clusters_to_gb(clust_obj, vocab_vec::Vector{String}, dist_mat::AbstractArray, coord_mat::AbstractArray, k::Int)
    cluster_labels = cutree(clust_obj, k = k)
    sils = 0 .- silhouettes(cluster_labels, dist_mat)
    df = DataFrame(
        "vocab" => vocab_vec,
        "cluster" => cluster_labels,
        "umap_x" => coord_mat[1, :],
        "umap_y" => coord_mat[2, :],
        "sil" => sils
        )
    gb = combine(
        groupby(df, "cluster"), 
        ["umap_x" , "umap_y"] => mean_eucl,
        ["umap_x" , "umap_y", "sil"] .=> mean, 
        "vocab" => (x) -> Ref(x)
        )
    rename!(gb, "umap_x_umap_y_mean_eucl" => "eucl", "umap_x_mean" => "centroid_x", "umap_y_mean" => "centroid_y", "sil_mean" => "sil", "vocab_function" => "vocab")
    gb[!, "vocab"] = Vector.(gb[!, "vocab"])
    gb[!, "sil"] = standardize(UnitRangeTransform, gb[!, "sil"])
    gb[!, "eucl"] = standardize(UnitRangeTransform, gb[!, "eucl"])
    out = [df, gb]
    return out
end

# classify
function generate_classifiers(x::Matrix, y::Vector{Int}, rf_num_rounds::Int, xg_num_rounds::Int, nn_epochs::Int)
    emb_dim = size(x, 2)
    label_dim = length(Set(y))
    layers = Chain(
        Flux.Dense(emb_dim => 512, Flux.relu),
        Flux.Dense(512 => 512),
        Flux.Dense(512 => label_dim),
        Flux.softmax
        )
    nn_builder = MLJFlux.@builder(layers)
    rfc = MLJ.@load(EvoTreeClassifier, pkg = "EvoTrees", verbosity = 0)(nrounds = rf_num_rounds, batch_size = 64)
    xgc = MLJ.@load(XGBoostClassifier, pkg = "XGBoost", verbosity = 0)(objective = "multi:softmax", num_round = xg_num_rounds)
    nnc = MLJ.@load(NeuralNetworkClassifier, pkg = "MLJFlux", verbosity = 0)(builder = nn_builder, optimiser = Flux.Adam(), epochs = nn_epochs, batch_size = 64)
    out = [rfc, xgc, nnc]
    return out 
end

function train_classifier(classifier::Union{EvoTreeClassifier, XGBoostClassifier, NeuralNetworkClassifier}, x::Matrix, y::Vector{Int})
    y = coerce(y, OrderedFactor)
    x = DataFrame(x, :auto)
    mach = machine(classifier, x, y, scitype_check_level = 0)
    @time out = MLJ.fit!(mach, verbosity = 0)
    return out 
end

function test_classifier(classifier::Machine, x::Matrix, y::Vector{Int})
    preds = predict_mode(classifier, x)
    acc = mean(preds .== y)
    out = [preds, acc]
    return out
end

# training 
function empty_res_df()
    out = DataFrame(
        "dataset" => String[],
        "k" => Int64[],
        "metric" => String[],
        "threshold" => Float64[],
        "bow" => String[],
        "classifier" => String[],
        "acc" => Float64[]
        )
    return out
end

function data_dict_to_classifiers(dd::Dict, k::Int, thresh::Float64)
    out_df = empty_res_df()
    for metric in ["eucl", "sil"]
        @info "Condensing features with metric $metric < $thresh, cluster cut $k"
        feat_dict = combine_features(dd["gb_$k"][!, "vocab"], dd["gb_$k"][!, metric], thresh, [])
        feat_docs = replace_features(dd["docs"], feat_dict)
        for (bow, bow_name) in zip([TfidfTransformer(), CountTransformer(), BM25Transformer()], ["tfidf", "count", "bm25"])
            @info "Encoding with $bow_name"
            mat = encode_bow(bow, feat_docs)
            vocab = pull_vocab(feat_docs)
            (xtrain, xtest), (ytrain, ytest) = MLJ.partition((mat, dd["int"]), 0.8, shuffle = true, multi = true)
            rfc, xgc, nnc = generate_classifiers(xtrain, ytrain, 100, 50, 50)
            for (classifier, classifier_name) in zip([rfc, xgc, nnc], ["rfc", "xgc", "nnc"])
                @info "$classifier_name classifier training"
                result = train_classifier(classifier, xtrain, ytrain)
                @info "$classifier_name classifier testing"
                acc = test_classifier(result, xtest, ytest)[2]
                push!(out_df, [dd["name"], k, metric, thresh, bow_name, classifier_name, acc])
                @info "completed eval loop with cluster cut at k = $k, metric = $metric, threshold = $thresh, bow = $bow_name, classifier = $classifier_name, test accuracy = $acc" 
            end
        end
    end
    return out_df
end

function data_dict_baselines(dd::Dict)
    out_df = empty_res_df()
    docs = dd["docs"]
    for (bow, bow_name) in zip([TfidfTransformer(), CountTransformer(), BM25Transformer()], ["tfidf", "count", "bm25"])
        @info "Encoding with $bow_name"
        mat = encode_bow(bow, docs)
        vocab = pull_vocab(docs)
        (xtrain, xtest), (ytrain, ytest) = MLJ.partition((mat, dd["int"]), 0.8, shuffle = true, multi = true)
        rfc, xgc, nnc = generate_classifiers(xtrain, ytrain, 100, 50, 50)
        for (classifier, classifier_name) in zip([rfc, xgc, nnc], ["rfc", "xgc", "nnc"])
            @info "$classifier_name classifier training"
            result = train_classifier(classifier, xtrain, ytrain)
            @info "$classifier_name classifier testing"
            acc = test_classifier(result, xtest, ytest)[2]
            push!(out_df, [dd["name"], 0, "_", 0.0, bow_name, classifier_name, acc])
            @info "completed eval loop with bow = $bow_name, classifier = $classifier_name, test accuracy = $acc"
        end
    end
    return out_df
end