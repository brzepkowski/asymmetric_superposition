# Bayes' rule and the posterior $P(c \mid s)$ of the appendix

The appendix of `lw_post.md` writes the posterior probability of a case $c$ given the reading $s$ as

$$ P(c \mid s) = \frac{P(c)\, \rho_c(s)}{\sum_{c'} P(c')\, \rho_{c'}(s)}. $$

This is the familiar

$$ P(A \mid B) = \frac{P(B \mid A)\, P(A)}{P(B)} $$

with $A = c$ (the case) and $B =$ "the reading is $s$", plus one wrinkle: $s$ is a continuous quantity, so the probability of "the reading is exactly $s$" is zero for every case, and densities take the place of probabilities.

## Term by term

| in $P(A \mid B) = \dfrac{P(B \mid A)\, P(A)}{P(B)}$ | in the appendix |
|---|---|
| $P(A)$, the prior of the hypothesis | $P(c)$, the prior probability of the case: $(1-p)^2$, $p(1-p)$, $p(1-p)$, $p^2$ |
| $P(B \mid A)$, the likelihood of the evidence under the hypothesis | $\rho_c(s)$, the density of the reading at $s$ *given* the case, e.g. uniform $1$ on $(0, 1]$ for "feature 1 alone", the triangle $1 - \lvert s \rvert$ on $[-1, 1]$ for "both" |
| $P(B)$, the total probability of the evidence | $\rho(s) = \sum_{c'} P(c')\, \rho_{c'}(s)$, the overall density of readings at $s$, by the law of total probability (a sum over the mutually exclusive cases) |
| $P(A \mid B)$, the posterior | $P(c \mid s)$ |

## Density as area

A density is a probability per unit of $s$: the probability of the reading falling in an interval is the area under the density over that interval. So $\int \rho_c(s)\, ds = 1$ for every case, because $\rho_c(s)$ is a *conditional* density: it describes where the reading lands given that case $c$ has already happened. Inside that world, the reading is certain to land somewhere, and "somewhere" is the whole $s$-axis, so the total probability, i.e. the total area, is $1$.

## Why densities are allowed

For a small interval around $s$,

$$ P(s \in [s, s + ds] \mid c) = \rho_c(s)\, ds, \qquad P(s \in [s, s + ds]) = \rho(s)\, ds. $$

Put these into Bayes' rule and the $ds$ cancels:

$$ P(c \mid s) = \frac{\rho_c(s)\, ds \cdot P(c)}{\rho(s)\, ds} = \frac{P(c)\, \rho_c(s)}{\sum_{c'} P(c')\, \rho_{c'}(s)}. $$

So the denominator of the appendix is just $P(B)$ written out via total probability. This is also why the appendix can say "dividing by the sum of the weights turns them into probabilities": the numerators $P(c)\, \rho_c(s)$ are the joint densities $\rho(s, c)$ of reading and case, and their sum over the cases is the marginal density $\rho(s)$.

## A worked value

At $s = 0.5$ with $p = 0.2$:

- likelihoods: $\rho_{1}(0.5) = 1$ (feature 1 alone), $\rho_{13}(0.5) = 1 - 0.5 = 0.5$ (both), and $0$ for the two cases that cannot produce a positive reading;
- priors: $P(1) = 0.16$, $P(13) = 0.04$;
- $P(B) \to \rho(0.5) = 0.16 \cdot 1 + 0.04 \cdot 0.5 = 0.18$;
- posteriors: $P(\text{feature 1 alone} \mid 0.5) = 0.16 / 0.18 \approx 0.89$, $P(\text{both} \mid 0.5) = 0.02 / 0.18 \approx 0.11$.

The decoder then returns $0.89 \cdot 0.5 + 0.11 \cdot 0.75 \approx 0.53$, the weighted average of the two cases' best guesses ($s$ and $(1+s)/2$).

## The point at $s = 0$

With neither feature active the reading is exactly $s = 0$, so that case has no density along $s$ but a point mass: $\rho_{\text{neither}}(s) = \delta(s)$, the Dirac delta, infinite at $0$ and zero elsewhere. In the ratio above this gives $P(\text{neither} \mid 0) = 1$ and $P(\text{neither} \mid s) = 0$ for every $s \neq 0$, consistent with $\hat{x}_1(0) = 0$ and with the other cases sharing all the weight away from the origin.
