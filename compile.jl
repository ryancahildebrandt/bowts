#!/usr/bin/env julia
# -*- coding: utf-8 -*- 
# Created on Fri Apr 21 04:41:08 PM EDT 2023 
# author: Ryan Hildebrandt, github.com/ryancahildebrandt

# imports
include("readin.jl")
include("utils.jl")


res_df = empty_res_df()
for k in [10, 25, 50]
    for t in [0.1:0.1:0.5...]
        temp = data_dict_to_classifiers(bitext_dd, k, t)
        append!(res_df, temp)
    end
end
res_df
CSV.write("outputs/bitext_df.csv", res_df)

res_df = empty_res_df()
for k in [10, 25, 50]
    for t in [0.1:0.1:0.5...]
        temp = data_dict_to_classifiers(title_dd, k, t)
        append!(res_df, temp)
    end
end
res_df
CSV.write("outputs/title_df.csv", res_df) 

res_df = empty_res_df()
for k in [10, 25, 50]
    for t in [0.1:0.1:0.5...]
        temp = data_dict_to_classifiers(abstract_dd, k, t)
        append!(res_df, temp)
    end
end
res_df
CSV.write("outputs/abstract_df.csv", res_df)

CSV.write("outputs/bitext_baseline_df.csv", data_dict_baselines(bitext_dd))
CSV.write("outputs/title_baseline_df.csv", data_dict_baselines(title_dd))
CSV.write("outputs/abstract_baseline_df.csv", data_dict_baselines(abstract_dd))

CSV.write("outputs/bitext_baseline_3000_df.csv", data_dict_baselines(bitext_dd))
CSV.write("outputs/title_baseline_3000_df.csv", data_dict_baselines(title_dd))
CSV.write("outputs/abstract_baseline_3000_df.csv", data_dict_baselines(abstract_dd))
