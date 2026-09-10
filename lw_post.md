# Introduction

The current, incredible performance of AI models can be partially attributed to their compression capabilities. This compression is imposed on them by the architectural choices made by engineers. For example, GPT-2 had a vocabulary of 50,257 tokens, yet its "operational space" was only of size 768. In such a space only 768 directions can be described fully independently, so the model had to find strategies to efficiently store all 50,257 input tokens in this reduced space. In general, we call the phenomenon in which a model stores more features than it has dimensions **superposition**. However, superposition comes at a cost. At a given step of computation, models utilizing it have neurons that fire for multiple different inputs (so-called polysemantic neurons), which makes them hard to interpret.

One theoretical way around this problem would be to train models with no superposition at all. However, this is believed to greatly reduce their performance. Instead, we would like to better understand the phenomenon itself.

Prior work, [*Toy Models of Superposition*](https://transformer-circuits.pub/2022/toy_model/index.html) (Elhage et al., 2022), on two-layer ReLU networks with independent input features of equal importance and sparsity, suggests that the best geometries for storing the features are uniform polyhedra. In this work we study a similar problem, but with models involving a dedicated encoding layer followed by one or more multilayer perceptrons (MLPs) with a bilinear activation. Current frontier LLMs also use a dedicated encoder as the first step of their computation, so we believe this setup can serve as good groundwork for a better understanding of larger models.

Knowing the exact definition of the data-generating process, we give a theoretical solution to the problem and demonstrate that uniform polyhedra are not the best possible solutions: different, non-symmetric geometries perform better. Surprisingly, only deep models are capable of exploiting these better strategies, while shallow models default to the symmetric ones. We give evidence that the previous results, which suggest uniform geometries are the best for the stated task, stem from the architectural choices made by the engineers rather than from the task itself.

# Setup

We will be working with the following toy setup:
- The input to the models consists of 4 independent features of equal importance. Each feature is active with a small probability $p$ (in our experiments we use $p = 0.2$), so the distribution of the features can be formally described as

$$ x_i = b_i u_i, \quad b_i \sim \mathrm{Bernoulli}(p), \quad u_i \sim \mathcal{U}[0, 1]. $$

- The models consist of a linear encoder layer with a bottleneck compressing the input features to a 2D plane, followed by one or more MLPs with a bilinear activation function:

$$ \hat{x} = \mathrm{MLP}(\dots \mathrm{MLP}(\mathrm{Enc}(x)) \dots). $$

- The task is to minimize the mean squared error (MSE) of the model's reconstruction:

$$ \mathcal{L}_{MSE} = \mathbb{E}_x \Bigg[ \frac{1}{n} \sum_{i=1}^n (\hat{x}_i - x_i)^2 \Bigg]. $$

As will become clear later, for the purposes of interpretation it is useful to divide the models under study into two parts: we will refer to the first, linear layer as the encoder and to the subsequent MLPs as the decoder. This division will also help us derive the theoretical reconstruction floor imposed by the architectural constraints (going from a 4D space down to 2D and back to 4D). The model under study and the division point are shown below. The 2D plane at this point is where all the geometries presented throughout this work live.

![The model under study: a linear encoder compressing 4 features to 2, followed by a stack of bilinear MLPs (the decoder).](figures/setup_diagram.png)

For completeness, we note that *Toy Models of Superposition* used a similar architecture on this very task. For 4 features compressed to a 2D space, it consists of a linear encoder given by a $2 \times 4$ matrix $W$, followed by a decoder that applies the transpose of the same matrix, $W^T$, a bias, and finally a ReLU activation. Because the same weight matrix $W$ is used in both layers, we will refer to this case as ***tied ReLU***.

# Theoretical solution

In this section we will introduce a simple, symmetric geometry that a theoretical encoder can implement, and provide a closed-form description of a decoder whose goal is to reconstruct the network's input from the point in the 2D plane that the encoder maps it to.

We will then present two strategies the encoder can employ to lower the reconstruction error, and finally demonstrate how these two techniques can be used in tandem to obtain the true best solution. This solution will serve as the theoretical reconstruction floor against which we will later compare the behavior of the actual models under study.

## Symmetrical antipodal encoding

The simplest way for the encoder to arrange the embeddings of the 4 input features on the 2D plane is as two antipodal pairs. The embeddings of each pair lie on a common line, pointing in opposite directions. The two lines are orthogonal, which allows the pairs to be analyzed independently, and their orientation is otherwise arbitrary. Below we show an example, where we have simply chosen the basis axes as the pair lines. Here $f_i \in \mathbb{R}^2$ denotes the **embedding** of feature $i$, the $i$-th column of the encoder matrix, so that the encoder's output, the **reading** seen by the decoder, is $\mathrm{Enc}(x) = \sum_{i=1}^4 x_i f_i$.

![The antipodal cross: the embeddings of the 4 input features placed as two antipodal pairs on orthogonal axes.](figures/naive_cross.png)

Let us focus for now on the horizontal line segment between $f_3$ and $f_1$; a reading on it is described by a single coordinate $s = x_1 - x_3 \in [-1, 1]$. We will use this segment to understand the interaction between the introduced encoder and a theoretical decoder through two lenses: (i) knowing the distribution of the features exactly, and (ii) having access only to an "oracle" that gives us samples of the data with no description of the underlying distribution. In both analyses we assume that the loss function, the MSE defined above, is known.

### Closed-form decoder from the known distribution

Knowing the distribution of the input features, we can distinguish four cases that result in a reading on this segment (as a reminder, we use the probability of a feature being active $p = 0.2$ throughout this work):
1. neither feature active, with probability $(1-p)^2 = 0.64$,
2. feature 1 alone active, with probability $p(1-p) = 0.16$,
3. feature 3 alone active, with probability $(1-p)p = 0.16$,
4. both features 1 and 3 active, with probability $p^2 = 0.04$.

This encoding method has one huge caveat. In the fourth case, with both features co-active, the reading can land anywhere on the segment (except for the endpoints $f_1$ and $f_3$ themselves). Because of that, even a perfect decoder cannot distinguish this case with full confidence from one in which only a single feature, or none, was active.

![The four cases on the segment between $f_3$ and $f_1$: feature 1 alone and features 1 and 3 together can produce the same reading.](figures/naive_cases.png)

Because of this information loss the decoder needs to average between different plausible scenarios to give the best possible response (measured with MSE): the best guess at a reading is the average of the guesses each scenario would make, weighted by how likely each scenario is to have produced that reading. The Appendix makes this precise.

Below we present the formula for such a theoretical decoder reconstructing the first feature, when given a reading at position $s$ on the segment (the derivation of this formula is presented in the Appendix).

$$ \hat{x}_1(s) = \mathbb{E}[x_1 \mid s] = \begin{cases} \dfrac{p\,(1+s)^2}{2\,(1+ps)} & -1 \le s < 0, \\[2ex] 0 & s = 0, \\[1ex] \dfrac{2(1-p)\,s + p\,(1-s^2)}{2\,(1-ps)} & 0 < s \le 1. \end{cases} $$


![The best decoder for $x_1$ on the antipodal pair, as a function of the reading on the segment between $f_3$ and $f_1$. Open circle: the one-sided limits $p/2$ at $s = 0$; closed circle: the value $0$ at $s = 0$ itself.](figures/posterior_x1_sym.png)

The decoder for the third feature is simply the mirror image of the curve above, $\hat{x}_3(s) = \hat{x}_1(-s)$, so we can picture the two decoders on the same plot:

![The best decoders for $x_1$ (blue) and $x_3$ (orange) on the antipodal pair, as functions of the reading $s$.](figures/posterior_pair_sym.png)

### Decoder from data samples alone

The above derivation was possible only because we knew the underlying distribution of the features. Now, let us try to obtain similar results while having access only to an "oracle" that gives us raw samples of the data (which is closer to a real-world situation).

Having samples of the data, we can also visualize them along our line segment:

![4096 samples of the pair drawn from the sampling process, plotted as the reading $s$ against the value $x_1$ the decoder has to predict.](figures/samples_x1_sym.png)

The above plot can make it more explicit that without the case of co-active features the best possible solution would be to use a ReLU function.

Now, how can we accommodate for the fact that there are situations in which different features are co-active? To do that we need to look at the definition of the loss function we are using.

In our setup the loss is the MSE. Let's consider all the samples that produce the same reading $s$. The decoder has to output one number for all of them, and its performance is measured by the squared distance from that number to each sample's true $x_1$. The number that minimizes the total squared distance to a set of values is their arithmetic mean, so the best decoder outputs the mean of $x_1$ over the samples with reading $s$.

So, the choice of the mean thus comes from the loss, not from the data. If we chose the mean absolute error instead, the best output would be the median of the same set of values, and with a 0–1 loss its most frequent value (the mode).

However, with a finite number samples we cannot take the mean over "all samples with reading $s$" literally, because no two samples have the same reading. Instead we split the line segment into very small intervals (bins), gather the samples whose reading falls into each, and average their $x_1$. The result is a step function approximating the closed-form curve, and it converges to it as we take more samples and narrower bins. Below we present the result of such an average computed on the data samples shown above:

![Binned means of the 4096 samples (100 bins) against the closed-form decoder $\hat{x}_1(s)$. The blue dot at $s = 0$ is the mean over the samples with reading exactly $0$.](figures/binned_x1_sym.png)

Having introduced the two methods of obtaining the best possible decoder depending on the context we are in, we now move on to the strategies, which can be applied to modify the initial encoder and increase the performance of the model as a whole.

## Asymmetrical antipodal encoding

The first modification is very simple: we make one of the embeddings shorter than the other.

![The symmetric antipodal pair and its asymmetric version, in which $f_1$ is shortened and $f_3$ lengthened ($|f_3| / |f_1| = 4.25$).](figures/levers.png)

This has a rather counterintuitive outcome. With this change the prediction error is "poured" into the reconstruction of a single feature; however, measured over both features, the total error turns out to be smaller!

Below we compare the two encodings. In the asymmetric one, $f_1$ is shortened and $f_3$ lengthened so that $|f_3|$ is $4.25$ times $|f_1|$. The top two panels show the closed-form decoders for both features, and the bottom one the MSE these decoders achieve, split into the contributions of the two features.

![Symmetric vs asymmetric antipodal pair ($|f_3| / |f_1| = 4.25$): the closed-form decoders for $x_1$ and $x_3$, and the MSE of the best decoder split into the two features' contributions.](figures/asym_compare.png)

Is it then always better to make the pair more asymmetric? To check this, we sweep the length ratio $|f_3| / |f_1|$ from $1$ to $100$ and compute, for each value, the MSE of the best decoder, both in closed form and from binned samples. In the binned case the segment is divided into bins of equal width and every feature gets its own decoder, which predicts the feature's mean value over the samples that land in the same bin. In both cases, as in the previous figure, we report the sum of the two features' MSEs rather than their average.

![The summed MSE of $x_1$ and $x_3$ under the best decoder as a function of the length ratio $|f_3| / |f_1|$. Top: the closed-form decoder, in total and split into the two features' contributions. Bottom: the closed form against binned decoders of three resolutions, one decoder per feature on the same bins, their MSEs summed. Triangles mark the minima. The dashed line marks the ratio $4.25$ used above.](figures/asym_sweep.png)

The closed-form curve first falls, because the error of the lengthened feature vanishes faster than the error of the shortened one grows, but it flattens out at a ratio of about $3$–$4$ (a minimum of $0.0106$ at $3.4$, against $0.0117$ for the symmetric pair) and then slowly rises again, toward $0.0113$: beyond this point the shortened feature is essentially unreadable whenever its partner is active, and its error dominates the total. Asymmetry therefore helps only up to a moderate ratio, and the optimum is shallow. A binned decoder has its minimum at the same ratio ($3.2$–$3.6$ for the three resolutions), but is stricter beyond it: once the short embedding becomes comparable to the width of a bin, the readings of the short feature on its own are no longer resolved, and the MSE climbs steeply, the earlier the coarser the bins (with $50$ bins the symmetric pair is better again above a ratio of $13$, with $400$ bins above $89$).

## "Opening" the antipodal pairs

Another modification boils down to slightly tilting one of the embeddings of a pair with respect to its partner. With this strategy the case of no active features still lands at the origin, and each feature active on its own still lands on its embedding vector, but the co-active case moves off the line into the interior of the parallelogram spanned by the two embeddings.

![Closing vs opening the pair: on the antipodal pair (left) a co-active reading lands on the line, where a single active feature could have produced it; on the opened pair (right) it lands inside the parallelogram, where nothing else can.](figures/opening.png)

This allows the decoder to distinguish features that are active on their own from co-active ones, which results in better reconstruction performance. In fact, a decoder of infinite resolution would be able to distinguish the co-activation at an arbitrarily small opening angle, since the co-active readings leave the line as soon as $\varepsilon > 0$.

However, trained models have finite resolution. Below we investigate its effects by studying what a binned decoder is actually capable of. We keep the length ratio of both pairs equal to $1$, tilt $f_3$ off the antipode of $f_1$ by $\varepsilon$, keep the other pair closed, and score every $\varepsilon$ with binned decoders of three resolutions. The bins are now square cells covering the 2D plane of readings, and again every feature gets its own decoder, which predicts the feature's mean value over the samples that land in the same cell. Since all four features are now involved, we report the MSE as defined in the Setup, i.e. averaged over the four features.

![The MSE of the binned decoder as a function of the opening angle $\varepsilon$ of the pair $(f_1, f_3)$, the pair $(f_2, f_4)$ closed, averaged over the four features, for three resolutions, one decoder per feature on the same cells (triangles mark the minima).](figures/opening_sweep.png)

The MSE has its minimum at a finite opening, which moves toward zero as the bins get finer. But why does the MSE not stay constant once $\varepsilon$ surpasses this threshold for a given binning? This is because opening one pair makes it interfere with the readings of the other pair. We will study this in more depth in the next section.

## Search for the best strategy

We close this section by investigating what is in fact the best strategy for a linear encoder followed by a decoder that is restricted to no particular function class and is limited only by the resolution of the bins used to compute it.

We do this by a free search over the encoder geometry. For an unconstrained decoder the angle between the two pair lines does not matter, so we may pin two of the embeddings to the axes and let the remaining two be arbitrary vectors in the plane, described by their angle and length. Every geometry is scored by the MSE of a decoder obtained numerically, as in "Decoder from data samples alone" but with two-dimensional bins. The four parameters are optimized by a random search followed by local descent.

Below we present the geometry obtained by the above search along with regions indicating the co-occurrence of various feature pairs.

![The geometry found by the search, with a zoom on the origin as an inset: the pair $(f_1, f_3)$ opened by $20.7^\circ$ at length ratio $13.5$, the pair $(f_2, f_4)$ by $20.4^\circ$ at $13.0$. Each shaded parallelogram is the region where the readings of two co-active features land; the two thin orange slivers along the axes belong to the pairs.](figures/optimal_geometry.png)

The search finds that the two free embeddings settle almost antipodal to the pinned ones, much shorter than them, and slightly tilted, each toward the long embedding of the other pair. The $(f_1, f_3)$ pair is opened by $20.7^\circ$ with a length ratio $|f_3| / |f_1| = 13.5$, and the $(f_2, f_4)$ pair by $20.4^\circ$ with $|f_4| / |f_2| = 13.0$, reaching an MSE floor of $0.00221$ per feature, against $0.0056$ for the closed symmetric pairs decoded at the same resolution ($96 \times 96$ bins).

Surprisingly, the length ratio found by the search is far larger than the $3$–$4$ that was optimal for a closed pair, for the binned decoders just as for the closed form. However, this is not a contradiction. The limiting factor in the closed pair case was that the two readings still occupied the same 1D line segment, so whenever two features were co-active, their contributions were mixed together. The moment one of the features is tilted, the co-active case moves into the interior of the parallelogram and can be distinguished by the decoder. But, as mentioned before, this comes at the price of interfering with the features of the second pair, and the best solution of finite resolution accommodates for that by shrinking the feature even more.

To check that this is indeed what sets the ratio, we repeat the asymmetry sweep on the opened geometry: both pairs opened by the angles found by the search ($20.7^\circ$ and $20.4^\circ$), the length ratio of both swept from $1$ to $100$, and every value scored by the same binned decoders as in the previous section. If shrinking the short embeddings really pays for the interference, the minimum should now lie well beyond the $3$–$4$ of the closed pair, and should move further out as the bins get finer, since it is the resolution that stops the shrinking. This is what we find: the minimum sits at a ratio of about $4.5$ with $24$ bins per axis, $5.6$ with $48$ and $6.3$ with $96$. For the finest binning, the MSE stays within $5\%$ of its minimum up to a ratio of $13$, which explains the result obtained by the search.

The MSE of the individual features (middle panel) shows the two sides of the trade-off. At a ratio of $1$ the co-active parallelograms are wide and overlap, which makes the value of a long feature ambiguous: this interference is the $0.0046$ of $x_3$ and $x_4$. As the tilted embeddings shrink, the parallelograms collapse onto the axes and this error falls to a floor of $0.0002$ by a ratio of about $13$. The short features $x_1$ and $x_2$, unlike in the closed pair, hardly pay for shrinking, because the decoder reads them off the line: their error rises to $0.0055$ by a ratio of $2$, then stays flat until the embeddings become comparable to a bin, and only then climbs steeply. The minimum of the average lies where the long features have gained the most while the short ones are still on their plateau.

We also checked whether the claim from the "Asymmetrical antipodal encoding" section still holds, i.e., whether asymmetry helps within the pair itself, this time with opened embeddings. In the bottom panel we show the sum of the MSEs of the two features of each pair. It turns out that the asymmetry pays off far more than for the closed pair: the MSE of $x_1 + x_3$ drops from $0.0091$ to $0.0055$, and rises again only at the resolution limit.

![The MSE of the binned decoders with both pairs opened by the angles found by the search ($20.7^\circ$ and $20.4^\circ$), as a function of the length ratio. Top: averaged over the four features, for three resolutions (triangles mark the minima). Middle: the MSE of each feature on its own and their average, with $96$ bins per axis. Bottom: the MSE summed over the two features of each pair, as in the closed-pair sweep. The dashed line marks the ratio $13$ found by the search.](figures/ratio_sweep.png)

To summarize, the best strategy needs to balance the following three mechanisms:
1. **Making the embeddings of a pair asymmetrical**, which pours most of the error into the prediction of one of them, which simultaneously reduces the overall MSE.
2. **Opening the pairs**, which makes it possible to distinguish the co-activation of two features from only one of them, or none, being active.
3. **Shortening the tilted embeddings**, which minimizes the interference with the features of the second pair.


# A model's strategy depends on its depth

Knowing what a good strategy is under the general constraints, let us now turn to what the trained models actually do. Below we present representative geometries of models with one and with four bilinear layers.

![Representative encoder geometries of trained models with one bilinear layer (left) and four bilinear layers (right). Embeddings of the same antipodal pair share a colour.](figures/gallery.png)

To make sure the above results are not an accident, we conducted 20 training runs for each depth, from 1 to 4 bilinear MLP layers, as well as for the tied ReLU model, and checked which strategy each trained model uses. We used sample of size 4,096 and 20,000 training steps in each training.

| model | symmetric pairs | asymmetric pairs | asymmetric + opened pairs | other |
|---|---|---|---|---|
| tied ReLU | 19 | 0 | 0 | 1 |
| 1 bilinear layer | 19 | 0 | 0 | 1 |
| 2 bilinear layers | 0 | 16 | 2 | 2 |
| 3 bilinear layers | 0 | 5 | 14 | 1 |
| 4 bilinear layers | 0 | 4 | 10 | 6 |

The "other" column gathers the runs whose encoder does not decompose into two antipodal pairs, either because some feature has no roughly opposite partner or because a feature is dropped entirely. It would also collect encoders whose pairs are opened (tilted beyond a small threshold) without being asymmetric, but no run produced such a geometry. Moreover, since opening never appears on its own, the table reports it only jointly with asymmetry, in the "asymmetric + opened pairs" column.

These results clearly show that models with a single bilinear MLP default to symmetric antipodal pairs, and so does the tied ReLU model. Only models with more bilinear MLPs start to use the strategies introduced in the previous section. Why is that? It turns out to boil down to the expressivity of the function class each model can implement.

We can understand this using the benchmark from the previous section.

## The symmetric vs. asymmetric trade-off

Let us first focus on the trade-off between symmetric and asymmetric pairs. We can plot the unconstrained decoders $\hat{x}_i(s)$ for both choices of the feature embeddings and approximate them with functions $\hat{x}_i'(s)$ of different complexity, corresponding to the model classes we are studying.

The decoder implemented by a model with a single bilinear MLP is a quadratic function. As we can see below, it cannot match the asymmetric benchmark properly, so the model defaults to the symmetric one, which results in a smaller MSE.

![A model with a single bilinear MLP, whose decoder is a quadratic in the reading: the best quadratic fit $\hat{x}_i'(s)$ to the unconstrained decoders $\hat{x}_i(s)$ of the symmetric and the asymmetric pair, and its MSE per feature against the unconstrained floor (dashed).](figures/class_quadratic.png)

The decoder of a model with four bilinear layers is far more expressive, because it is a polynomial of degree 16. Such a function can properly approximate both the symmetric and the asymmetric embedding, so the model chooses the asymmetric one, which results in a smaller loss.

![A model with four bilinear MLPs, whose decoder is a polynomial of degree 16 in the reading: the same comparison.](figures/class_deg16.png)

Finally, the tied ReLU model faces the same challenge as the model with a single bilinear MLP. It cannot approximate the asymmetric case well, so it also settles on the symmetric solution.

![Tied ReLU model: the same comparison.](figures/class_relu_tied.png)

## Detecting the opening of a pair

We have found that the symmetric vs. asymmetric trade-off comes down to the expressivity of the function class each model can implement. A similar argument explains why some models open their pairs while others do not.

As before, we fix the encoder, this time with one pair slightly opened: $f_3$ is tilted by $25^\circ$ away from the antipode of $f_1$, while the other pair stays closed. The readings of co-active features 1 and 3 now fill the thin parallelogram between $f_1$ and $f_3$, shown in orange below, whereas all other co-active pairs fill the much larger light-blue ones.

![One pair opened: $f_3$ is tilted by $25^\circ$ off the antipode of $f_1$. The orange parallelogram collects the readings of co-active features 1 and 3. The dashed line is the cut along which we examine the decoder, and the highlighted segment where it crosses the orange region is the strip.](figures/opened_geometry.png)

To see what such an encoding demands of the decoder, we cut the plane along a line crossing the orange parallelogram and follow the reconstruction of feature 3 along the cut. Outside the strip, a reading on the cut can only come from other combinations of features, so the best decoder reports feature 3 as inactive. Inside the strip, feature 3 is co-active with feature 1, and the decoder should raise its estimate accordingly. The unconstrained decoder is therefore a narrow spike over the strip.

Below we compare it with the best fits from the function classes of our models. The tied ReLU, quadratic and quartic decoders barely react to the strip at all. The degree-8 decoder of the model with three bilinear MLPs responds with a bump, but a broad one that spills far beyond the strip, so it captures only part of the benefit. This is nevertheless enough for opening to pay off, which is consistent with the table above, where models with three bilinear MLPs are the first to open their pairs in most runs. Only the degree-16 decoder of the model with four bilinear MLPs approximates the spike well. Opening a pair therefore brings no benefit to shallow models, which is why they keep their pairs closed, while deeper models can read the strip and profit from it.

![The reconstruction of feature 3 along the cut: the unconstrained decoder (grey) spikes over the strip. The tied ReLU stays at zero, the quadratic and quartic classes barely react, the degree-8 class responds with a broad bump, and only the degree-16 class approximates the spike.](figures/opened_decoded.png)

# How well do the models approximate the best decoder?

We have given indications of why the models may or may not use different strategies to improve their performance in superposition. But how close do the trained models actually come to these theoretical results?

For this comparison we retrain the two representative models of the previous section (the same seeds, hence the same initializations) on a much larger sample of $2^{20}$ inputs instead of the 4,096 used there, keeping the 20,000 training steps.

Below we present the binned decoders for all four features, which the model with a single bilinear MLP is supposed to reconstruct.

![Binned decoder ($2^{20}$ samples, $40 \times 40$ bins over the reading plane), single bilinear MLP, seed 10.](figures/binned3d_bilinear1.png)

Next, we show the best quadratic approximation of these binned decoders.

![Best decoder of the class (least-squares quadratic), single bilinear MLP, seed 10.](figures/class3d_bilinear1.png)

Finally, we show the difference between what the model has actually learned and this best decoder of its class. For the model with a single bilinear MLP the difference is surprisingly small!

![Trained model minus the best decoder of its class, single bilinear MLP, seed 10.](figures/class_diff_bilinear1.png)

The analogous results for the model with four bilinear MLPs are presented below. In this case the difference between the trained model and the best fit in its class is much larger than for the single-MLP model.

![Binned decoder, four bilinear MLPs, seed 9.](figures/binned3d_bilinear4.png)

![Best decoder of the class (least-squares polynomial of degree 16), four bilinear MLPs, seed 9.](figures/class3d_bilinear4.png)

![Trained model minus the best decoder of its class, four bilinear MLPs, seed 9.](figures/class_diff_bilinear4.png)

To take a closer look, we cut the reading plane of each model along the line of its longest embedding and follow the reconstruction of that feature along the cut. The shallow model traces the best quadratic exactly, while the deeper model visibly departs from the best polynomial of degree 16. Here, too, the difference between the deeper model and the best solution in its class is larger than for the shallow one.

![Cut along the line of the longest embedding: the binned decoder, the best decoder of the class and the trained model.](figures/cut_long.png)

Finally, to quantify the above observations, we define the *MSE gap* $= (\mathrm{MSE}_{\text{model}} - \mathrm{MSE}_{\text{class}}) / \mathrm{MSE}_{\text{model}}$, i.e., the fraction of the model's error that it could still remove by becoming the best decoder of its class. It is equal to zero when the model already is that decoder.

| model | MSE, binned decoder | MSE, best of the class | MSE, trained model | MSE gap |
|---|---|---|---|---|
| single bilinear MLP, 20,000 steps | 0.0061 | 0.0093 | 0.0093 | 0.0% |
| four bilinear MLPs, 20,000 steps | 0.0036 | 0.0041 | 0.0060 | 31.9% |

These results underline once more that the shallow model is essentially the best approximation of the binned decoder that its class allows, while the deeper model comes close to it, but not as close as the shallow one.

Is the remaining gap merely the result of a training run that is too short? To check this, we continued the training: starting from the 20,000-step checkpoint of the table above, we trained the model further on the same sample, with a smaller learning rate so that it refines the solution it has already found rather than jumping to a completely new one. The smaller learning rate does not freeze the geometry, though. Over the continuation it keeps slowly drifting within the same strategy, the pairs opening from about $19^\circ$ and $21^\circ$ to $27^\circ$ and $44^\circ$ and the length ratio of one of them growing from $2.6$ to $6.5$. As before, each checkpoint is therefore compared with the best decoder of its class for the geometry it has at that moment.

| model | MSE, binned decoder | MSE, best of the class | MSE, trained model | MSE gap |
|---|---|---|---|---|
| four bilinear MLPs, 20,000 steps | 0.0036 | 0.0041 | 0.0060 | 31.9% |
| continued to 150,000 steps | 0.0038 | 0.0040 | 0.0050 | 19.9% |
| continued to 300,000 steps | 0.0039 | 0.0040 | 0.0049 | 18.6% |
| continued to 600,000 steps | 0.0039 | 0.0040 | 0.0049 | 18.1% |

The MSE gap shrinks, but clearly saturates. Still, one might argue that the culprit is the small learning rate itself: perhaps the model is stuck refining a mediocre solution that a bolder training would escape. To check this, we also retrained the model from scratch, from the same initialization, with the original larger learning rate spread over 150,000, 300,000 and 600,000 steps. Because the learning rate is annealed over the course of training, stretching the run changes it at every step, so each of these runs follows a completely different trajectory and lands in a different solution. Indeed, the resulting MSE gaps change non-monotonically with the training length — $18.0\%$, $27.1\%$ and $21.6\%$ — but even the best of them merely matches the $18\%$ at which the continuation saturates, and none improves on it.

The remaining gap of about $18\%$ is therefore not a matter of training length, but of some different phenomenon. We leave it as an open question, since the main purpose of this work was to show that deeper models use asymmetry to their advantage. Nevertheless, understanding why they do not approach the best solutions in their class the way the shallow models do would be a valuable direction for future work.

# Discussion



## Limitations and future work

1. Our method divides the model into two parts: the first, linear layer is the encoder, and the remaining MLPs form the decoder. We do not yet have a proper understanding of what the individual MLP layers are doing. We believe that feeding the whole geometry output by one layer into the following one, and tracking how it is transformed, would also shed light on the role of the individual layers.

2. For explanatory reasons, the results shown in this work were limited to a 2D plane at the encoder's output. We would like to check whether our findings survive in higher-dimensional bottlenecks with more features: do the embeddings still organize into asymmetric, slightly tilted antipodal pairs, and do deeper decoders still profit from them? Since such geometries can no longer be simply plotted, this will require replacing pictures with quantitative measures, such as the length ratios and tilts of the pairs and the interference between them.

3. We found that, contrary to their shallow counterparts, deeper models do not converge to the best possible decoder in their class. We would like to understand precisely why deeper models stop short: for example, whether the obstacle lies in the optimization itself, or in the way the stacked bilinear layers parametrize the polynomials they can in principle express.

# Summary

We have demonstrated that uniform polyhedra are not inherently the best solution to the task of reconstructing independent features of equal importance and sparsity. The earlier indications that they might stem from the architectural choices made by the engineers rather than from the task itself.

We have shown two strategies that models can employ to improve on the simple symmetric antipodal embedding of the features: making the embeddings of a pair asymmetric and slightly tilting one of them. Only deeper models can use these strategies, because only their expressivity allows for it.

Finally, we have shown that shallow models are essentially the best possible approximations of the theoretical binned decoders within their class, while deeper models diverge slightly from the best approximation in theirs. This divergence is not an artifact of a training run that is too short: continuing the training of the deeper model shrinks the gap from $32\%$ to about $18\%$ of its error, where it saturates, and retraining it from scratch, with the original larger learning rate annealed over the longer run, changes the gap non-monotonically and does no better. Why deeper models stop short of the best decoders of their class remains an open question.

# Acknowledgements

BR would like to thank [Pivotal](https://www.pivotal-research.org/) for their support. This research was carried out during the Pivotal AI Safety Research Fellowship.

# Appendix: derivation of the closed-form decoder

We derive $\hat{x}_1(s) = \mathbb{E}[x_1 \mid s]$ for the symmetric antipodal pair, where $s = x_1 - x_3$. Recall that each feature is inactive (equal to $0$) with probability $1 - p$ and otherwise uniformly distributed on $[0, 1]$, independently of the other.

The best guess at a reading $s$ is the average of $x_1$ over all the ways of producing that reading, each way weighted by how likely it is. This is the averaging between scenarios of the main text, and it can be written down exactly. Let $c$ stand for the case (neither feature active, feature 1 alone, feature 3 alone, both), $P(c)$ for its prior probability and $\rho_c(s)$ for the density of the readings it produces at $s$. By Bayes' rule, the posterior probability of a case given the reading is

$$ P(c \mid s) = \frac{P(c)\, \rho_c(s)}{\sum_{c'} P(c')\, \rho_{c'}(s)}. $$

The weight of a case is how often it produces the reading $s$: its prior probability $P(c)$ times the density $\rho_c(s)$ of its readings at $s$. Dividing by the sum of the weights turns them into probabilities. The best guess is then the average of the best guesses of the individual cases, weighted by these posterior probabilities:

$$ \hat{x}_1(s) = \mathbb{E}[x_1 \mid s] = \sum_c P(c \mid s)\, \mathbb{E}[x_1 \mid s, c]. $$


The figure below shows these weights and the per-case guesses. One case needs a remark: with neither feature active the reading is exactly $s = 0$, so this case has no density along $s$ but a point mass, $P(c)\, \rho_c(s) = (1-p)^2\, \delta(s)$ with $\delta$ the Dirac delta. At $s = 0$ it therefore outweighs every other case, and at any $s \neq 0$ it has no weight at all. The rest of this appendix computes the weights and takes the weighted average.

![Left: the weight of each case, $P(c)\, \rho_c(s)$, i.e. how often it produces the reading $s$; the case of neither feature active produces only $s = 0$, so its weight is a Dirac delta, $(1-p)^2\, \delta(s)$, drawn as an arrow in the usual way (its height is not to scale; the number next to it is the probability it carries). Right: the best guess of each case, $\mathbb{E}[x_1 \mid s, c]$ (for the co-active case, the average over the pairs $(x_1, x_3)$ producing $s$), and their weighted average, which is the closed-form decoder $\hat{x}_1(s)$.](figures/appendix_cases.png)

For each case we need $P(c)$, $\rho_c(s)$ and $\mathbb{E}[x_1 \mid s, c]$:

- **Neither active**: $P(c) = (1-p)^2$. The reading is exactly $s = 0$, so $\rho_c(s) = \delta(s)$, and $\mathbb{E}[x_1 \mid s, c] = 0$. This case alone produces $s = 0$, hence $\hat{x}_1(0) = 0$; the other three produce readings $s \neq 0$.
- **Feature 1 alone**: $P(c) = p(1-p)$. The reading $s = x_1$ is uniform on $(0, 1]$, so $\rho_c(s) = 1$ there, and $\mathbb{E}[x_1 \mid s, c] = s$.
- **Feature 3 alone**: $P(c) = p(1-p)$. The reading $s = -x_3$ is uniform on $[-1, 0)$, so $\rho_c(s) = 1$ there, and $\mathbb{E}[x_1 \mid s, c] = 0$.
- **Both active**: $P(c) = p^2$. Since $x_1$ and $x_3$ are independent and uniform, the pair $(x_1, x_3)$ is spread evenly over the unit square $[0, 1]^2$ of its possible values (the figure below, left; this is a picture of the input values, not of the reading plane), and the reading $s = x_1 - x_3$ is constant along each diagonal line of that square. How often a reading occurs is therefore proportional to the length of its diagonal: it is longest for $s = 0$ (the main diagonal from $(0, 0)$ to $(1, 1)$) and shrinks linearly to a single corner at $s = \pm 1$ (one feature at $1$, the other at $0$). This gives the triangle $\rho_c(s) = 1 - |s|$ on $[-1, 1]$ (right).

![Left: with both features active, the pair $(x_1, x_3)$ is uniform on the unit square of its possible values, and each reading $s$ corresponds to one diagonal line $x_1 - x_3 = s$ of it. Right: the length of that diagonal, as a function of $s$, is the density $\rho_c(s) = 1 - |s|$ of the co-active case.](figures/appendix_square.png)

It remains to find the best guess of the co-active case, $\mathbb{E}[x_1 \mid s, c]$, the average of $x_1$ over all the pairs $(x_1, x_3)$ that produce the reading $s$, i.e. over the diagonal $x_1 - x_3 = s$. Given $s$, $x_1$ is uniformly distributed over its range on the diagonal, and the mean of a uniform variable is the midpoint of its range. For $s > 0$ the diagonal runs from $(x_1, x_3) = (s, 0)$ to $(1, 1 - s)$, so $x_1$ is uniform on $[s, 1]$ with mean $\frac{s + 1}{2}$. For $s < 0$ it runs from $(0, |s|)$ to $(1 + s, 1)$, so $x_1$ is uniform on $[0, 1 + s]$ with mean $\frac{1 + s}{2}$. In both cases $\mathbb{E}[x_1 \mid s, c] = \frac{1 + s}{2}$.

With $P(c)$, $\rho_c(s)$ and $\mathbb{E}[x_1 \mid s, c]$ of every case in hand, we can now evaluate the weighted average. Only the cases that can produce the reading enter it, which leaves two ranges of $s$ to consider.

For $0 < s \le 1$ the reading can come from "feature 1 alone" or from "both":

$$ \hat{x}_1(s) = \frac{p(1-p)\, s + p^2 (1-s)\, \frac{1+s}{2}}{p(1-p) + p^2 (1-s)} = \frac{2(1-p)\, s + p\,(1 - s^2)}{2\,(1 - ps)}. $$

For $-1 \le s < 0$ it can come from "feature 3 alone", which contributes $x_1 = 0$, or from "both":

$$ \hat{x}_1(s) = \frac{p^2 (1+s)\, \frac{1+s}{2}}{p(1-p) + p^2 (1+s)} = \frac{p\,(1+s)^2}{2\,(1 + ps)}. $$

Together with $\hat{x}_1(0) = 0$ this is the formula of the main text.
