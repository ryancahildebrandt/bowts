#!/usr/bin/env julia
# -*- coding: utf-8 -*- 
# Created on Mon Jun 12 07:00:08 PM EDT 2023 
# author: Ryan Hildebrandt, github.com/ryancahildebrandt

# imports
using GLM
using CSV

include("utils.jl")

#max available results
results_max = empty_res_df()
append!(results_max, CSV.read("outputs/abstract_3000_df.csv", DataFrame))
append!(results_max, CSV.read("outputs/abstract_baseline_3000_df.csv", DataFrame))
append!(results_max, CSV.read("outputs/title_10000_df.csv", DataFrame))
append!(results_max, CSV.read("outputs/title_baseline_10000_df.csv", DataFrame))
append!(results_max, CSV.read("outputs/bitext_df.csv", DataFrame))
append!(results_max, CSV.read("outputs/bitext_baseline_df.csv", DataFrame))

combine(groupby(results_max, "threshold"), "acc" => mean)
filt = (results_max[!, "bow"] .== "count") .& (results_max[!, "classifier"] .== "xgc")
combine(groupby(results_max[filt, ["dataset", "bow", "classifier", "acc"]], ["dataset", "bow", "classifier"]), "acc" => mean)

max_mains = DataFrame(coeftable(fit(LinearModel, @formula(acc ~ dataset + k + metric + threshold + bow + classifier), results_max)))
max_metric_interactions = DataFrame(coeftable(fit(LinearModel, @formula(acc ~ k * metric * threshold), results_max)))
max_model_interactions = DataFrame(coeftable(fit(LinearModel, @formula(acc ~ dataset * bow * classifier), results_max)))

#3000 sample results
results_sample = empty_res_df()
append!(results_sample, CSV.read("outputs/abstract_3000_df.csv", DataFrame))
append!(results_sample, CSV.read("outputs/abstract_baseline_3000_df.csv", DataFrame))
append!(results_sample, CSV.read("outputs/title_3000_df.csv", DataFrame))
append!(results_sample, CSV.read("outputs/title_baseline_3000_df.csv", DataFrame))
append!(results_sample, CSV.read("outputs/bitext_3000_df.csv", DataFrame))
append!(results_sample, CSV.read("outputs/bitext_baseline_3000_df.csv", DataFrame))

combine(groupby(results_max, "bow"), "acc" => mean)
filt = (results_sample[!, "bow"] .== "count") .& (results_sample[!, "classifier"] .== "rfc")
combine(groupby(results_sample[filt, ["dataset", "bow", "classifier", "acc"]], ["dataset", "bow", "classifier"]), "acc" => mean)

sample_mains = DataFrame(coeftable(fit(LinearModel, @formula(acc ~ dataset + k + metric + threshold + bow + classifier), results_sample)))
sample_metric_interactions = DataFrame(coeftable(fit(LinearModel, @formula(acc ~ k * metric * threshold), results_sample)))
sample_model_interactions = DataFrame(coeftable(fit(LinearModel, @formula(acc ~ dataset * bow * classifier), results_sample)))

CSV.write("outputs/results_sample_df.csv", results_sample)
CSV.write("outputs/results_max_df.csv", results_max)