using FileIO, DelimitedFiles, NPZ, JLD2


save(
    "20news-sub.jld2", 
    Dict(
        "D" => 1_000, "V" => 668, 
        "w" => npzread("w.npy"), "doc" => npzread("doc.npy"), 
        "vocab" => vec(readdlm("vocab.csv", ',', String, '\n'))
    ),
)