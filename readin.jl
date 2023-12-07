#!/usr/bin/env julia
# -*- coding: utf-8 -*- 
# Created on Sat Apr  8 08:26:11 PM EDT 2023 
# author: Ryan Hildebrandt, github.com/ryancahildebrandt

# imports
using DataFrames
using CSV
using JLD2
using FileIO

include("utils.jl")

# embedding file 
_vocab = []
_embeddings = []
# for a limited vocab
#file_iterator = Iterators.take(eachline("data/glove.6B.50d.txt"), 100000)
file_iterator = eachline("data/glove.6B.50d.txt")

for l in file_iterator
    line = split(l)
    push!(_vocab, line[1])
    push!(_embeddings, parse.(Float32, line[2:end]))
end
push!(_vocab, "OOV")
push!(_embeddings, zeros(50))

# embedding objects
ind_dict = Dict(word => ind for (ind, word) in enumerate(_vocab))
emb_df = DataFrame(
    "term" => _vocab, 
    "emb" => _embeddings
    )

# bitext dataset
#https://github.com/bitext/customer-support-intent-detection-training-dataset
bitext_df = CSV.read("data/20000-Utterances-Training-dataset-for-chatbots-virtual-assistant-Bitext-sample/20000-Utterances-Training-dataset-for-chatbots-virtual-assistant-Bitext-sample/20000-Utterances-Training-dataset-for-chatbots-virtual-assistant-Bitext-sample.csv", DataFrame)[:, ["utterance", "intent"]]
rename!(bitext_df, "utterance" => "text", "intent" => "label")
bitext_df[!, "int"] = str_to_int(bitext_df[!, "label"])

# multilabel classification from analytics vidhya hackathon
#https://www.kaggle.com/datasets/shivanandmn/multilabel-classification-dataset?select=train.csv
avh_df = CSV.read("data/avh_train.csv", DataFrame)
avh_df = stack(avh_df, 4:9)
rename!(avh_df, "variable" => "label")
avh_df[!, "int"] = str_to_int(avh_df[!, "label"])

title_df = avh_df[avh_df[!, "value"] .== 1, ["TITLE", "label", "int"]]
rename!(title_df, "TITLE" => "text")

abstract_df = avh_df[avh_df[!, "value"] .== 1, ["ABSTRACT", "label", "int"]]
rename!(abstract_df, "ABSTRACT" => "text")

#bitext_df = bitext_df[sample(1:nrow(bitext_df), 20000), :] #21535
bitext_dd = build_data_dict(bitext_df, "bitext")
JLD2.save("data/bitext_dd.jld2", "data", bitext_dd)
#bitext_dd = JLD2.load("data/bitext_dd.jld2")["data"]

#title_df = title_df[sample(1:nrow(title_df), 10000), :] #309328
title_dd = build_data_dict(title_df, "title")
JLD2.save("data/title_dd.jld2", "data", title_dd)
#title_dd = JLD2.load("data/title_dd.jld2")["data"]

#abstract_df = abstract_df[sample(1:nrow(abstract_df), 3000), :] #309328
abstract_dd = build_data_dict(abstract_df, "abstract")
JLD2.save("data/abstract_dd.jld2", "data", abstract_dd)
#abstract_dd = JLD2.load("data/abstract_dd.jld2")["data"]

@info "Readin Complete"
