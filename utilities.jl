using PyCall
import PyPlot
const plt = PyPlot
const wordcloud = pyimport("wordcloud")

function extract_expected_topic_parameters(chain, D, K, V, nsamples, nthins)
    local fmt
    if Symbol("θ[1,1]") in keys(chain)
        fmt = (i, j) -> "[$i,$j]"
    else Symbol("θ[:,1][1]") in keys(chain)
        fmt = (i, j) -> "[:,$j][$i]"
    end
    β = hcat([[mean(chain["β$(fmt(i,j))"][end-nsamples:nthins:end]) for i in 1:V] for j in 1:K]...)
    θ = hcat([[mean(chain["θ$(fmt(i,j))"][end-nsamples:nthins:end]) for i in 1:K] for j in 1:D]...)
    return (β=β, θ=θ)
end

function plot_word_cloud(ax, freqs)
    wc = wordcloud.WordCloud(background_color="white", width=800, height=600)
    wc = wc.generate_from_frequencies(freqs)
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
end

function make_word_clouds_plot(beta, vocab; topk=50)
    num_rows = div(size(beta, 2), 5)
    fig, axes = plt.subplots(num_rows, 5, figsize=(16, 2.5 * num_rows))
    for n in 1:size(beta, 2)
        indices = sortperm(beta[:,n]; rev=true)
        words = map(i -> vocab[i], indices)
        freqs = Dict(vocab[i] => beta[i,n] for i in indices)

        i, j = divrem(n - 1, 5)
        ax = axes[i+1,j+1]
        plot_word_cloud(ax, freqs)
        ax.set_title("Topic $n")
    end
    plt.close(fig)
    return fig
end